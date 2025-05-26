from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int

from turban.utils.util import agg_fast_to_slow, get_chunking_index
from turban.utils.spectra import power_spectrum


def process_level3(
    shear: Float[ndarray, "n_shear time_fast"],
    pspd: Float[ndarray, "time_fast"],
    segment_length: int,
    segment_overlap: int,
    diss_length: int,
    diss_overlap: int,
    sampling_freq: float,
    spatial_response_wavenum: float,
    freq_highpass: float,
    section_number: Int[ndarray, "time_fast"],
) -> tuple[
    Float[ndarray, "time_slow k"],  # k
    Float[ndarray, "n_shear time_slow wavenumber"],  # Pk
    Float[ndarray, "n_shear time_slow wavenumber"],  # Pf
    Float[ndarray, "wavenumber"],  # freq
    Float[ndarray, "time_slow"],  # pspda
    Int[ndarray, "time_slow"],  # section_number_slow
]:
    ii = get_chunking_index(
        shear.shape[-1],
        segment_length,
        segment_overlap,
        diss_length,
        diss_overlap,
        section_number,
    )

    Pf, freq = power_spectrum(shear, sampling_freq, reshape_index=ii)

    # platform speed
    pspda = agg_fast_to_slow(pspd, reshape_index=ii)

    section_number_slow = section_number[..., ii].max(axis=-1).max(axis=-1)

    # to wavenumber domain
    Pk = Pf * pspda[newaxis, :, newaxis] / segment_length / (sampling_freq / 2)
    k: Float[ndarray, "time_slow k"] = freq[newaxis, :] / pspda[:, newaxis]

    # apply corrections
    if False:
        correction_factor_spatial = apply_compensation_spatial_response(
            Pk, k, spatial_response_wavenum
        )
        _ = apply_compensation_highpass(Pk, freq, freq_highpass)
    # apply_removal_coherent_vibrations(P)

    apply_var_conserve(Pk, k, shear, ii)

    return k, Pk, Pf, freq, pspda, section_number_slow


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


def apply_var_conserve(
    Pk: Float[ndarray, "n_shear time_slow waveno"],
    k: Float[ndarray, "time_slow k"],
    shear: Float[ndarray, "n_shear time_fast"],
    reshape_index: Int[ndarray, "diss_chunk fft_chunk segment_length"],
) -> Float[ndarray, 'n_shear time_slow']:
    dk = k[..., 1] - k[..., 0]
    varPk = Pk[..., 1:].sum(axis=-1) * dk[newaxis, :] # disregard first wavelength
    sr = shear[:, reshape_index]
    srf = sr.reshape(sr.shape[:-2] + (sr.shape[-2]*sr.shape[-1],)) # flatten last two axes
    srf = np.ascontiguousarray(srf) # performance enhancement for np.var
    corr_factor = np.var(srf, axis=-1) / varPk
    Pk *= corr_factor[..., newaxis]
    return corr_factor
