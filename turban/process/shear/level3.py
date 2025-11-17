from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int

from turban.utils.util import agg_fast_to_slow, get_chunking_index
from turban.utils.spectra import power_spectrum


def process_level3(
    shear: Float[ndarray, "n_shear time_fast"],
    senspeed: Float[ndarray, "time_fast"],
    segment_length: int,
    segment_overlap: int,
    chunk_length: int,
    chunk_overlap: int,
    sampfreq: float,
    spatial_response_wavenum: float,
    freq_highpass: float,
    section_number: Int[ndarray, "time_fast"],
) -> tuple[
    Float[ndarray, "time_slow k"],  # k
    Float[ndarray, "n_shear time_slow waveno"],  # Pk
    Float[ndarray, "n_shear time_slow waveno"],  # Pf
    Float[ndarray, "waveno"],  # freq
    Float[ndarray, "time_slow"],  # senspeeda
    Int[ndarray, "time_slow"],  # section_number_slow
]:
    ii = get_chunking_index(
        section_number,
        (chunk_length, chunk_overlap),
        (segment_length, segment_overlap),
    )

    psi_f, freq = power_spectrum(shear, sampfreq, reshape_index=ii)

    # platform speed
    senspeeda = agg_fast_to_slow(senspeed, reshape_index=ii)

    section_number_slow = section_number[..., ii].max(axis=-1).max(axis=-1)

    # to waveno domain
    psi_k = psi_f * senspeeda[newaxis, :, newaxis] / segment_length / (sampfreq / 2)
    waveno: Float[ndarray, "time_slow k"] = freq[newaxis, :] / senspeeda[:, newaxis]

    # apply corrections
    if False:
        correction_factor_spatial = apply_compensation_spatial_response(
            psi_k, waveno, spatial_response_wavenum
        )
        _ = apply_compensation_highpass(psi_k, freq, freq_highpass)
    # apply_removal_coherent_vibrations(P)

    apply_var_conserve(psi_k, waveno, shear, ii)

    return waveno, psi_k, psi_f, freq, senspeeda, section_number_slow


def apply_compensation_spatial_response(
    psi_k: Float[ndarray, "n_shear time_slow k"],
    waveno: Float[ndarray, "time_slow k"],
    waveno_0: float,
) -> Float[ndarray, "time_slow k"]:
    correction_factor = 1.0 + (waveno / waveno_0) ** 2
    # TODO Eqn. 18 text: Do not use spectra where correction exceeds 10
    correction_factor[correction_factor > 10.0] = 10.0  # dirty hack
    psi_k *= correction_factor[newaxis, :, :]
    return correction_factor


def apply_compensation_highpass(
    psi_f: Float[ndarray, "n_shear time_slow f"],
    freq: Float[ndarray, "f"],
    freq_highpass: float,
) -> Float[ndarray, "f"]:
    correction_factor = (1.0 + (freq_highpass / freq) ** 2.0) ** 2.0
    psi_f *= correction_factor[newaxis, :]
    return correction_factor


def apply_var_conserve(
    psi_k: Float[ndarray, "n_shear time_slow waveno"],
    waveno: Float[ndarray, "time_slow k"],
    shear: Float[ndarray, "n_shear time_fast"],
    reshape_index: Int[ndarray, "diss_chunk fft_chunk segment_length"],
) -> Float[ndarray, "n_shear time_slow"]:
    dk = waveno[..., 1] - waveno[..., 0]
    varPk = psi_k[..., 1:].sum(axis=-1) * dk[newaxis, :]  # disregard first wavelength
    sr = shear[:, reshape_index]
    srf = sr.reshape(
        sr.shape[:-2] + (sr.shape[-2] * sr.shape[-1],)
    )  # flatten last two axes
    srf = np.ascontiguousarray(srf)  # performance enhancement for np.var
    corr_factor = np.var(srf, axis=-1) / varPk
    psi_k *= corr_factor[..., newaxis]
    return corr_factor
