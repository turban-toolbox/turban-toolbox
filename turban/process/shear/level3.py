from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int

from turban.utils.util import agg_fast_to_slow, fast_to_slow_reshape_index
from turban.utils.spectra import power_spectrum


def process_level3(
    shear: Float[ndarray, "n_shear time_fast"],
    pspd: Float[ndarray, "time_fast"],
    fft_length: int,
    fft_overlap: int,
    diss_length: int,
    diss_overlap: int,
    sampling_freq: float,
    spatial_response_wavenum: float,
    freq_highpass: float,
    section_marker: Int[ndarray, "time_fast"],
) -> tuple[
    Float[ndarray, "time_slow k"],  # k
    Float[ndarray, "n_shear time_slow wavenumber"],  # Pk
    Float[ndarray, "n_shear time_slow wavenumber"],  # Pf
    Float[ndarray, "wavenumber"],  # freq
    Float[ndarray, "time_slow"],  # pspda
    Int[ndarray, "time_slow"],  # section_marker_slow
]:
    ii = fast_to_slow_reshape_index(
        shear.shape[-1],
        fft_length,
        fft_overlap,
        diss_length,
        diss_overlap,
        section_marker,
    )

    Pf, freq = power_spectrum(shear, sampling_freq, reshape_index=ii)

    # platform speed
    pspda = agg_fast_to_slow(pspd, reshape_index=ii)

    section_marker_slow = section_marker[..., ii].max(axis=-1).max(axis=-1)

    # to wavenumber domain
    Pk = Pf * pspda[newaxis, :, newaxis] / fft_length / (sampling_freq / 2)
    k: Float[ndarray, "time_slow k"] = freq[newaxis, :] / pspda[:, newaxis]

    # apply corrections
    if False:
        correction_factor_spatial = apply_compensation_spatial_response(
            Pk, k, spatial_response_wavenum
        )
        _ = apply_compensation_highpass(Pk, freq, freq_highpass)
    # apply_removal_coherent_vibrations(P)

    # Ugly variance preserving procedure
    shear_reshape = shear[..., ii]  # reshape to fft length windows
    print('Shape shear orig', np.shape(shear))
    print('Shape shear', np.shape(shear_reshape))
    if True:
        for ishear in range(np.shape(Pk)[0]):    
            for isegment in range(np.shape(Pk)[1]):
                dk = k[ishear,1] - k[ishear,0]
                shear_flat = shear_reshape[ishear,isegment,:,:].flatten()
                #print('shear_flat', shear_flat)
                #print('shear_flat shape',np.shape(shear_flat))
                #print('Pk data',Pk[ishear,isegment,:])
                # In PkPk[ishear,isegment,0] is an inf, ignore that
                varPk = sum(Pk[ishear,isegment,1:]) * dk
                varshear = np.var(shear_flat)
                varscale = varshear / varPk
                #print('k',k)
                #print('dk',dk)
                #print('Shear',shear)
                #print('len Shear',np.shape(shear),sum(np.isnan(shear)),'sh',np.shape(Pk),'k',np.shape(k))
                #print('varpk',varPk,'varshear',varshear)
                #print('Varscale',varscale)
                Pk[ishear,isegment,:] *= varscale
                #break

    return k, Pk, Pf, freq, pspda, section_marker_slow


def apply_compensation_spatial_response(
    x: Float[ndarray, "n_shear time_slow k"],
    k: Float[ndarray, "time_slow k"],
    k0: float,
) -> Float[ndarray, "time_slow k"]:
    correction_factor = 1.0 + (k / k0) ** 2
    # TODO Eqn. 18 text: Do not use spectra where correction exceeds 10
    correction_factor[correction_factor > 10.0] = 10.0  # dirty hack
    x *= correction_factor[newaxis, :, :]
    return correction_factor


def apply_compensation_highpass(
    x: Float[ndarray, "n_shear time_slow f"],
    freq: Float[ndarray, "f"],
    freq_highpass: float,
) -> Float[ndarray, "f"]:
    correction_factor = (1.0 + (freq_highpass / freq) ** 2.0) ** 2.0
    x *= correction_factor[newaxis, :]
    return correction_factor
