import warnings
import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray, newaxis
from scipy.signal import butter, freqz, lfilter, lfiltic
from scipy.special import erf, gamma

from turban.utils.util import integrate, reshape_any_first, reshape_halfoverlap_last
from turban.utils.util import agg_fast_to_slow, get_chunking_index
from turban.utils.spectra import spectrum

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
    section_number: Int[ndarray, "time_fast"],
) -> tuple[
    Float[ndarray, "time_slow k"],  # k
    Float[ndarray, "ntemp time_slow waveno"],  # Pk
    Float[ndarray, "ntemp time_slow waveno"],  # Pf
    Float[ndarray, "waveno"],  # freq
    Float[ndarray, "time_slow"],  # senspeeda
    Int[ndarray, "time_slow"],  # section_number_slow
    Float[ndarray, "ntemp waveno"],  # psi_noise
]:
    ii = get_chunking_index(
        section_number,
        (chunk_length, chunk_overlap),
        (segment_length, segment_overlap),
    )

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
    senspeeda = agg_fast_to_slow(senspeed, reshape_index=ii)

    section_number_slow = section_number[..., ii].max(axis=-1).max(axis=-1)

    # double check spectral corrections
    psi_f *= correction_frequency_response_bilinear(freq=freq, Fs=sampfreq)
    psi_f *= correction_frequency_response_vachon_lueck(freq=freq, senspeed=senspeeda)

    waveno = freq / senspeeda
    psi_k: Float[ndarray, "time_slow freq"] = (
        psi_f * senspeeda / segment_length / (sampfreq / 2)
    )

    # to waveno domain
    psi_k = psi_f * senspeeda[newaxis, :, newaxis]
    waveno: Float[ndarray, "time_slow k"] = freq[newaxis, :] / senspeeda[:, newaxis]

    # TODO explore supplying own noise level by input arguments
    psi_noise = get_noise(psi_k)
    # TODO double check whether we should subtract noise here
    # Pk -= Pnoise

    return waveno, psi_k, psi_f, freq, senspeeda, section_number_slow, psi_noise


def correction_frequency_response_bilinear(
    freq: Float[ndarray, "time frequency"], Fs: float
):
    assert len(freq.shape) == 2
    assert freq.shape[0] == 1
    assert freq.shape[1] >= 1

    diff_gain = 1.5

    [b, a] = butter(
        1, 1 / (2 * np.pi * diff_gain * Fs / 2)
    )  #  The LP-filter that was applied
    w, junk = freqz(
        b, a, freq.shape[1], fs=Fs, include_nyquist=True
    )  # axis=1 indexes frequencies
    junk = np.absolute(junk) ** 2  #  The mag-squared of the applied LP-filter.
    H = 1 / (1 + (2 * np.pi * freq * diff_gain) ** 2)  #  What should have been applied

    assert np.all(np.allclose(w[newaxis, :], freq))
    bl_correction = H / junk  #  The bilinear transformation correction.
    return bl_correction


def correction_frequency_response_vachon_lueck(
    freq: Float[ndarray, "1 frequency"], senspeed: Float[ndarray, "time 1"]
) -> Float[ndarray, "time frequency"]:
    F_0 = 25 * np.sqrt(np.abs(senspeed))  # cutoff freq
    tau_therm = 1 / ((2 * np.pi * F_0) / np.sqrt(np.sqrt(2) - 1))  # time constant
    Hinv = (
        1 + (2 * np.pi * tau_therm * freq) ** 2
    ) ** 2  # inverse of the frequency response
    # - correction (Hinv is nondimensional so can apply directly to Pk_gradT)
    return Hinv


def correction_frequency_response():
    raise warnings.warn("To be implemented")
    return 1


def get_noise(
    spectra: Float[ndarray, "time frequency"],
) -> Float[ndarray, "frequency"]:
    """
    Define noise as average of least intense 5% of spectra
    """
    if spectra.shape[0] > 0:
        # a measure of the intensity of the spectrum
        spec_intens = np.mean(spectra[:, :20], axis=1)
        # 5 % least intense spectra
        (ii,) = np.where(spec_intens < np.percentile(spec_intens, 5))
        noise = 10 ** np.mean(np.log10(spectra[ii, :]), axis=0)
        return noise
    else:
        return np.zeros((spectra.shape[1],), dtype=float)
