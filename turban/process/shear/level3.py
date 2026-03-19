from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int

from turban.utils.util import agg_fast_to_slow, get_chunking_index
from turban.utils.spectra import spectrum
from turban.utils.logging import get_logger

logger = get_logger(__name__)

def process_level3(
    shear: Float[ndarray, "nshear time_fast"],
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
    Float[ndarray, "nshear time_slow waveno"],  # psi_k_sh
    Float[ndarray, "nshear time_slow waveno"],  # psi_f_sh
    Float[ndarray, "waveno"],  # freq
    Float[ndarray, "time_slow"],  # senspeeda
    Int[ndarray, "time_slow"],  # section_number_slow
]:
    """Compute shear power spectra and convert to wavenumber domain.

    Parameters
    ----------
    shear : ndarray, shape (nshear, time_fast)
        Despiked shear time series for each sensor.
    senspeed : ndarray, shape (time_fast,)
        Platform speed through water in m/s.
    segment_length : int
        Number of samples per FFT segment.
    segment_overlap : int
        Overlap between successive FFT segments in samples.
    chunk_length : int
        Number of samples per dissipation chunk.
    chunk_overlap : int
        Overlap between successive dissipation chunks in samples.
    sampfreq : float
        Sampling frequency in Hz.
    spatial_response_wavenum : float
        Spatial response cutoff wavenumber in cpm for sensor deconvolution.
    freq_highpass : float
        High-pass cutoff frequency in Hz for spectral correction.
    section_number : ndarray of int, shape (time_fast,)
        Section marker array; zero marks invalid data.

    Returns
    -------
    waveno : ndarray, shape (time_slow, k)
        Wavenumber array in cpm for each chunk.
    psi_k_sh : ndarray, shape (nshear, time_slow, waveno)
        Shear power spectral density in wavenumber domain.
    psi_f_sh : ndarray, shape (nshear, time_slow, waveno)
        Shear power spectral density in frequency domain.
    freq : ndarray, shape (waveno,)
        Frequency array in Hz.
    senspeeda : ndarray, shape (time_slow,)
        Chunk-averaged platform speed.
    section_number_slow : ndarray of int, shape (time_slow,)
        Section markers aggregated to slow time.
    """
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
    psi_k: Float[ndarray, "nshear time_slow k"],
    waveno: Float[ndarray, "time_slow k"],
    waveno_0: float,
) -> Float[ndarray, "time_slow k"]:
    """Apply spatial response compensation to wavenumber spectra in-place.

    Parameters
    ----------
    psi_k : ndarray, shape (nshear, time_slow, k)
        Shear spectra in wavenumber domain; modified in-place.
    waveno : ndarray, shape (time_slow, k)
        Wavenumber array in cpm.
    waveno_0 : float
        Sensor spatial response cutoff wavenumber in cpm.

    Returns
    -------
    ndarray, shape (time_slow, k)
        Correction factor applied to the spectra.
    """
    correction_factor = 1.0 + (waveno / waveno_0) ** 2
    # TODO Eqn. 18 text: Do not use spectra where correction exceeds 10
    correction_factor[correction_factor > 10.0] = 1
    psi_k *= correction_factor[newaxis, :, :]
    return correction_factor


def apply_compensation_highpass(
    psi_f: Float[ndarray, "nshear time_slow f"],
    freq: Float[ndarray, "f"],
    freq_highpass: float,
) -> Float[ndarray, "f"]:
    """Apply high-pass filter compensation to frequency spectra in-place.

    Parameters
    ----------
    psi_f : ndarray, shape (nshear, time_slow, f)
        Shear spectra in frequency domain; modified in-place.
    freq : ndarray, shape (f,)
        Frequency array in Hz.
    freq_highpass : float
        High-pass filter cutoff frequency in Hz.

    Returns
    -------
    ndarray, shape (f,)
        Correction factor applied to the spectra.
    """
    correction_factor = (1.0 + (freq_highpass / freq) ** 2.0) ** 2.0
    psi_f *= correction_factor[newaxis, :]
    return correction_factor
