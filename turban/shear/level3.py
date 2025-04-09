from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int

from turban.util import agg_fast_to_slow, fast_to_slow_reshape_index
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
    ancillary: dict[str, Float[ndarray, "time_fast"]] = None,  # average to time_slow
) -> tuple[
    Float[ndarray, "time_slow k"],  # k
    Float[ndarray, "n_shear time_slow wavenumber"],  # Pk
    Float[ndarray, "n_shear time_slow wavenumber"],  # Pf
    Float[ndarray, "wavenumber"],  # freq
    Float[ndarray, "time_slow"],  # pspda
    Int[ndarray, "time_slow"],  # section_marker_slow
    dict[str, Float[ndarray, "time_slow"]],  # ancillary_out
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

    # Average to time_slow
    data_fast: Float[ndarray, "variable time_fast"] = (
        pspd[newaxis, :]  # add dimension
        if ancillary is None
        else np.stack((pspd, *(arr for k, arr in ancillary.items())), axis=0)
    )
    # platform speed
    data_slow: Float[ndarray, "variable time_slow"] = agg_fast_to_slow(
        data_fast, reshape_index=ii
    )
    pspda = data_slow[0, :]

    section_marker_slow = section_marker[..., ii].max(axis=-1).max(axis=-1)

    # to wavenumber domain
    Pk = Pf * pspda[newaxis, :, newaxis] / fft_length / (sampling_freq / 2)
    k: Float[ndarray, "time_slow k"] = freq[newaxis, :] / pspda[:, newaxis]

    # apply corrections
    correction_factor_spatial = apply_compensation_spatial_response(
        Pk, k, spatial_response_wavenum
    )
    _ = apply_compensation_highpass(Pk, freq, freq_highpass)
    # apply_removal_coherent_vibrations(P)

    ancillary_out = (
        {
            name: (["time_slow"], data_slow[ind + 1, :])
            for ind, name in enumerate(ancillary.keys())
        }
        if ancillary is not None
        else {}
    )

    return k, Pk, Pf, freq, pspda, section_marker_slow, ancillary_out


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
