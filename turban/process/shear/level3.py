from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int

from turban.utils.util import agg_fast_to_slow, get_chunking_index
from turban.utils.spectra import spectrum


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

    psi_f, freq = spectrum(
        shear,
        sampfreq,
        section_number=section_number,
        chunk_length=chunk_length,
        chunk_overlap=chunk_overlap,
        segment_length=segment_length,
        segment_overlap=segment_overlap,
    )

    # platform speed
    senspeeda = agg_fast_to_slow(
        senspeed,
        section_number_or_data_len=section_number,
        chunk_length=chunk_length,
        chunk_overlap=chunk_overlap,
    )

    section_number_slow = section_number[..., ii].max(axis=-1).max(axis=-1)

    _ = apply_compensation_highpass(psi_f, freq, freq_highpass)

    # to waveno domain
    psi_k = psi_f * senspeeda[newaxis, :, newaxis]
    waveno: Float[ndarray, "time_slow k"] = freq[newaxis, :] / senspeeda[:, newaxis]

    _ = apply_compensation_spatial_response(psi_k, waveno, spatial_response_wavenum)

    # apply_removal_coherent_vibrations(P) # TODO

    return waveno, psi_k, psi_f, freq, senspeeda, section_number_slow


def apply_compensation_spatial_response(
    psi_k: Float[ndarray, "n_shear time_slow k"],
    waveno: Float[ndarray, "time_slow k"],
    waveno_0: float,
) -> Float[ndarray, "time_slow k"]:
    correction_factor = 1.0 + (waveno / waveno_0) ** 2
    # TODO Eqn. 18 text: Do not use spectra where correction exceeds 10
    correction_factor[correction_factor > 10.0] = 1
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
