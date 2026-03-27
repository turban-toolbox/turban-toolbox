import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray, newaxis
from scipy.signal import butter, freqz, lfilter, lfiltic
from scipy.special import erf, gamma

from turban.utils.util import integrate, reshape_any_first, reshape_halfoverlap_last
from turban.utils.util import agg_fast_to_slow, get_chunking_index
from turban.utils.spectra import spectrum

from turban.utils.logging import get_logger

logger = get_logger(__name__)

# nu, kin. viscosity of water; assumed known constant
viscosity_kinematic = 0.0000016
# molecular temperature diffusivity [m^2/s]
diffusivity_temp = 0.00000014
# constant for batchelor spectrum
q_b = 3.7


def temperature_gradient_spectra(
    dtempdt: Float[ndarray, "ntemp time_fast"],
    senspeed: Float[ndarray, "time_fast"],
    segment_length: int,
    segment_overlap: int,
    chunk_length: int,
    chunk_overlap: int,
    sampfreq: float,
    waveno_limit_upper: float,
    diff_gain: float,
    section_number: Int[ndarray, "time_fast"],
) -> tuple[
    Float[ndarray, "time_slow k"],  # k
    Float[ndarray, "ntemp time_slow waveno"],  # Pk
    Float[ndarray, "ntemp time_slow waveno"],  # Pf
    Float[ndarray, "waveno"],  # freq
    Float[ndarray, "time_slow"],  # senspeeda
    Int[ndarray, "time_slow"],  # section_number_slow
    Float[ndarray, "ntemp waveno"],  # psi_noise
    Int[ndarray, "time_slow n_chunks n_segments"],  # ii
]:
    """Compute temperature gradient power spectra and convert to wavenumber domain.

    Parameters
    ----------
    dtempdt : ndarray, shape (ntemp, time_fast)
        Time derivative of temperature for each sensor.
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
    waveno_limit_upper : float
        Upper wavenumber integration limit in cpm.
    diff_gain : float
        Time constant of the NTC high-pass pre-emphasis differentiator in seconds.
    section_number : ndarray of int, shape (time_fast,)
        Section marker array; zero marks invalid data.

    Returns
    -------
    waveno : ndarray, shape (time_slow, k)
        Wavenumber array in cpm for each chunk.
    psi_k : ndarray, shape (ntemp, time_slow, waveno)
        Temperature gradient spectra in wavenumber domain.
    psi_f : ndarray, shape (ntemp, time_slow, waveno)
        Temperature gradient spectra in frequency domain.
    freq : ndarray, shape (waveno,)
        Frequency array in Hz.
    senspeeda : ndarray, shape (time_slow,)
        Chunk-averaged platform speed.
    section_number_slow : ndarray of int, shape (time_slow,)
        Section markers aggregated to slow time.
    psi_noise : ndarray, shape (ntemp, waveno)
        Estimated noise spectrum from the least intense 5% of chunks.
    ii : ndarray of int, shape (time_slow, n_chunks, n_segments)
        Chunking index used to aggregate fast-to-slow quantities.
    """
    ii = get_chunking_index(
        section_number,
        (chunk_length, chunk_overlap),
        (segment_length, segment_overlap),
    )

    psi_f: Float[ndarray, "ntemp time_slow waveno"]
    psi_f, freq = spectrum(
        dtempdt,
        sampfreq,
        section_number=section_number,
        chunk_length=chunk_length,
        chunk_overlap=chunk_overlap,
        segment_length=segment_length,
        segment_overlap=segment_overlap,
    )

    # platform speed
    senspeeda: Float[ndarray, "time_slow"] = agg_fast_to_slow(
        senspeed, reshape_index=ii
    )

    section_number_slow = section_number[..., ii].max(axis=-1).max(axis=-1)

    # double check spectral corrections
    psi_f *= correction_frequency_response_bilinear(
        freq=freq, sampfreq=sampfreq, diff_gain=diff_gain
    )
    psi_f *= correction_frequency_response_vachon_lueck(freq=freq, senspeed=senspeeda)

    waveno: Float[ndarray, "time_slow k"] = freq[newaxis, :] / senspeeda[:, newaxis]
    psi_k: Float[ndarray, "ntemp time_slow freq"] = (
        psi_f * senspeeda[newaxis, :, newaxis] / segment_length / (sampfreq / 2)
    )

    # to waveno domain
    psi_k = psi_f * senspeeda[newaxis, :, newaxis]
    waveno: Float[ndarray, "time_slow k"] = freq[newaxis, :] / senspeeda[:, newaxis]

    # TODO explore supplying own noise level by input arguments
    psi_noise = get_noise(psi_k)
    # TODO double check whether we should subtract noise here
    # Pk -= Pnoise

    return waveno, psi_k, psi_f, freq, senspeeda, section_number_slow, psi_noise, ii


def correction_frequency_response_bilinear(
    freq: Float[ndarray, "frequency"],
    sampfreq: float,
    diff_gain: float,
):
    [b, a] = butter(
        1, 1 / (2 * np.pi * diff_gain * sampfreq / 2)
    )  #  The LP-filter that was applied
    w, junk = freqz(
        b, a, len(freq), fs=sampfreq, include_nyquist=True
    )  # axis=1 indexes frequencies
    junk = np.absolute(junk) ** 2  #  The mag-squared of the applied LP-filter.
    H = 1 / (1 + (2 * np.pi * freq * diff_gain) ** 2)  #  What should have been applied

    assert np.all(np.allclose(w[newaxis, :], freq))
    bl_correction = H / junk  #  The bilinear transformation correction.
    return bl_correction


def correction_frequency_response_vachon_lueck(
    freq: Float[ndarray, "frequency"], senspeed: Float[ndarray, "time"]
) -> Float[ndarray, "time frequency"]:
    F_0 = 25 * np.sqrt(np.abs(senspeed[:, newaxis]))  # cutoff freq
    tau_therm = 1 / ((2 * np.pi * F_0) / np.sqrt(np.sqrt(2) - 1))  # time constant
    Hinv = (
        1 + (2 * np.pi * tau_therm * freq[newaxis, :]) ** 2
    ) ** 2  # inverse of the frequency response
    # - correction (Hinv is nondimensional so can apply directly to Pk_gradT)
    return Hinv


def get_noise(
    spectra: Float[ndarray, "*any time frequency"],
) -> Float[ndarray, "*any frequency"]:
    """
    Define noise as average of least intense 5% of spectra
    """
    if spectra.shape[-2] > 0:
        # a measure of the intensity of the spectrum
        spec_intens: Float[ndarray, "*any time"] = np.mean(spectra[..., :20], axis=-1)
        # 5 % least intense spectra
        least_intense: Float[ndarray, "*any time frequency"] = np.where(
            spec_intens[..., newaxis] < np.percentile(spec_intens, 5), spectra, np.nan
        )
        noise = 10 ** np.nanmean(np.log10(least_intense), axis=-2)
        return noise
    else:
        return np.zeros((*spectra.shape[:-2], spectra.shape[-1]), dtype=float)
