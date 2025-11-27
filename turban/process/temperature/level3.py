import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray, newaxis
from scipy.signal import butter, freqz, lfilter, lfiltic
from scipy.special import erf, gamma

from turban.utils.util import integrate, reshape_any_first, reshape_halfoverlap_last

# nu, kin. viscosity of water; assumed known constant
viscosity_kinematic = 0.0000016
# molecular temperature diffusivity [m^2/s]
diffusivity_temp = 0.00000014
# constant for batchelor spectrum
q_b = 3.7


def temperature_gradient_spectra(
    dTdt: Float[ndarray, "time_fast"],
    senspeed: Float[ndarray, "time_fast"],
    chunklen: int,
    chunkoverlap: int,
    segment_length: int,
    sampfreq: float,
) -> tuple[
    Float[ndarray, "time_slow waveno"],
    Float[ndarray, "time_slow waveno"],
    Float[ndarray, "1 waveno"],
]:
    yr: Float[ndarray, "time_slow freq"] = reshape_halfoverlap_last(
        dTdt, segment_length
    )
    yr -= yr.mean(axis=1)[:, np.newaxis]
    yr *= np.hanning(segment_length)[np.newaxis, :]

    freq: Float[ndarray, "1 freq"] = np.fft.rfftfreq(segment_length, d=1 / sampfreq)[
        np.newaxis, :
    ]
    Fyr = np.fft.rfft(yr)[:, :]
    Pf: Float[ndarray, "time_slow freq"] = (Fyr.conj() * Fyr).real

    senspeeda: Float[ndarray, "time_slow 1"] = reshape_any_first(
        reshape_halfoverlap_last(senspeed, segment_length).mean(axis=1)[:, np.newaxis],
        chunklen,
        chunkoverlap,
    ).mean(axis=1)
    assert senspeeda.shape[1] == 1

    correction = correction_frequency_response_bilinear(
        freq=freq, Fs=sampfreq
    ) * correction_frequency_response_vachon_lueck(freq=freq, senspeed=senspeeda)

    # average spectra by chunks
    Pf = reshape_any_first(Pf, chunklen, chunkoverlap).mean(axis=1)
    # Pf = Pf * correction

    k = freq / senspeeda
    Pk: Float[ndarray, "time_slow freq"] = (
        Pf * senspeeda / segment_length / (sampfreq / 2)
    )

    Pnoise = get_noise(Pk)
    # Pk -= Pnoise

    return k, Pk, Pnoise


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
    return 1
