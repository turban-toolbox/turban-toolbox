from typing import Literal

import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray, newaxis

from turban.process.utemp.config import FP07_RESPONSE_MODELS, FP07ResponseModelName
from turban.utils.util import agg_fast_to_slow, get_chunking_index
from turban.utils.spectra import spectrum

from turban.utils.logging import get_logger

logger = get_logger(__name__)


def temperature_gradient_spectra(
    dtempdt: Float[ndarray, "ntemp time_fast"],
    senspeed: Float[ndarray, "time_fast"],
    segment_length: int,
    segment_overlap: int,
    chunk_length: int,
    chunk_overlap: int,
    sampfreq: float,
    section_number: Int[ndarray, "time_fast"],
    spectral_correction: Float[ndarray, "freq"] | None,
    tau0: float | None,
    gamma: float | None,
    npoles: int | None,
    response_model: FP07ResponseModelName | None = None,
    noise_model: Literal["empirical"] = "empirical",
) -> tuple[
    Float[ndarray, "time_slow waveno"],  # waveno
    Float[ndarray, "ntemp time_slow waveno"],  # psi_k
    Float[ndarray, "ntemp time_slow waveno"],  # psi_f
    Float[ndarray, "freq"],  # freq
    Float[ndarray, "time_slow"],  # senspeed
    Int[ndarray, "time_slow"],  # section_number_slow
    Float[ndarray, "ntemp waveno"],  # psi_noise
    Int[ndarray, "time_slow chunk_length"],  # ii
]:
    """Compute temperature gradient power spectra and convert to wavenumber domain.

    Parameters
    ----------
    dtempdt : ndarray, shape (ntemp, time_fast)
        Time derivative of temperature for each sensor in K s-1.
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
    section_number : ndarray of int, shape (time_fast,)
        Section marker array; zero marks invalid data.
    spectral_correction : ndarray, shape (freq,), or None
        Multiplicative correction of the frequency spectra (e.g. for instrument
        electronics), on the frequency grid of the spectra. None: no correction.
    tau0 : float or None
        FP07 time constant in s at a speed of 1 m s-1, see ``get_fp07_response``.
    gamma : float or None
        Exponent of the speed dependence of the FP07 time constant.
    npoles : int or None
        Number of poles of the FP07 response.
    response_model : str or None, optional
        Name of a built-in FP07 response model, takes priority over
        ``tau0``, ``gamma`` and ``npoles``. ``"none"``: no response correction.
    noise_model : {"empirical"}, optional
        Method to estimate the noise spectrum, see ``get_noise_empirical``.

    Returns
    -------
    waveno : ndarray, shape (time_slow, waveno)
        Wavenumber array in cpm for each chunk.
    psi_k : ndarray, shape (ntemp, time_slow, waveno)
        Temperature gradient spectra in wavenumber domain in K2 m-2 cpm-1,
        corrected for the FP07 response.
    psi_f : ndarray, shape (ntemp, time_slow, waveno)
        Spectra of ``dtempdt`` in frequency domain in K2 s-2 Hz-1.
    freq : ndarray, shape (freq,)
        Frequency array in Hz.
    senspeed : ndarray, shape (time_slow,)
        Chunk-averaged platform speed.
    section_number_slow : ndarray of int, shape (time_slow,)
        Section markers aggregated to slow time.
    psi_noise : ndarray, shape (ntemp, waveno)
        Estimated noise spectrum from the least intense 5% of chunks.
    ii : ndarray of int, shape (time_slow, chunk_length)
        Chunking index used to aggregate fast-to-slow quantities.
    """
    ii = get_chunking_index(
        section_number,
        (chunk_length, chunk_overlap),
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

    section_number_slow: Int[ndarray, "time_slow"] = agg_fast_to_slow(
        section_number, reshape_index=ii, agg_method="max"
    )

    if spectral_correction is not None:
        if len(spectral_correction) != len(freq):
            logger.warning(
                "Has spectral_correction been computed based on the same segment_length?"
            )
        psi_f *= spectral_correction

    # to waveno domain: dT/dx = (dT/dt) / U, and S(k) = U S(f), so psi_k = psi_f / U
    # TODO LATER we currently deal with dT/dt; ATOMIX-chi draft as of now specifies dT/dx at Level 1
    waveno: Float[ndarray, "time_slow waveno"] = (
        freq[newaxis, :] / senspeeda[:, newaxis]
    )
    psi_k: Float[ndarray, "ntemp time_slow waveno"] = (
        psi_f / senspeeda[newaxis, :, newaxis]
    )
    psi_k /= get_fp07_response(
        freq=freq,
        senspeed=senspeeda,
        tau0=tau0,
        gamma=gamma,
        npoles=npoles,
        response_model=response_model,
    )

    # TODO LATER explore supplying own noise level by input arguments
    psi_noise = get_noise_empirical(psi_k)

    return waveno, psi_k, psi_f, freq, senspeeda, section_number_slow, psi_noise, ii


def fp07_time_constant(
    senspeed: Float[ndarray, "time"],
    tau0: float,
    gamma: float,
) -> Float[ndarray, "time"]:
    """Time constant of the FP07 thermistor, tau = tau0 W^gamma.

    Parameters
    ----------
    senspeed : ndarray, shape (time,)
        Platform speed W in m s-1.
    tau0 : float
        Time constant in s at a speed of 1 m s-1.
    gamma : float
        Exponent of the speed dependence.

    Returns
    -------
    ndarray, shape (time,)
        Time constant in s.
    """
    return tau0 * np.abs(senspeed) ** gamma


def get_fp07_response(
    freq: Float[ndarray, "freq"],
    senspeed: Float[ndarray, "time"],
    tau0: float | None = None,
    gamma: float | None = None,
    npoles: int | None = None,
    model: FP07ResponseModelName | None = None,
) -> Float[ndarray, "time freq"]:
    """Power transfer function H(f) of the FP07 thermistor.

    H = (1 + (2 pi tau f)^2)^-n with n = ``npoles`` and tau = tau0 W^gamma (see
    ``fp07_time_constant``).

    Parameters
    ----------
    freq : ndarray, shape (freq,)
        Frequency in Hz.
    senspeed : ndarray, shape (time,)
        Platform speed in m s-1.
    tau0, gamma, npoles : float, float, int, optional
        Parameters of the response, required if ``model`` is None.
    model : str, optional
        Name of a model in ``FP07_RESPONSE_MODELS``, takes priority over
        ``tau0``, ``gamma`` and ``npoles``. ``"none"`` gives H = 1.

    Returns
    -------
    ndarray, shape (time, freq)
        Power transfer function.
    """
    if model is None and tau0 is None and gamma is None and npoles is None:
        logger.info("No response model selected")
        return np.ones((len(senspeed), len(freq)))
    if model is not None:
        logger.info(f"Selected FP07 reponse model `{model}`")
        res = FP07_RESPONSE_MODELS[model]
        tau0, gamma, npoles = res.tau0, res.gamma, res.npoles
    else:
        logger.info (f"Manually selected tau0: {tau0}, gamma: {gamma}, npoles {npoles}.")
    tau = fp07_time_constant(senspeed, tau0, gamma)
    return (1 + (2 * np.pi * tau[:, newaxis] * freq[newaxis, :]) ** 2) ** (-npoles)


# TODO LATER validate the use of this against get_fp07_response
def _correction_frequency_response_vachon_lueck(
    freq: Float[ndarray, "frequency"], senspeed: Float[ndarray, "time"]
) -> Float[ndarray, "time frequency"]:
    logger.warning("Deprecated")
    F_0 = 25 * np.sqrt(np.abs(senspeed[:, newaxis]))  # cutoff freq
    tau_therm = 1 / ((2 * np.pi * F_0) / np.sqrt(np.sqrt(2) - 1))  # time constant
    Hinv = (
        1 + (2 * np.pi * tau_therm * freq[newaxis, :]) ** 2
    ) ** 2  # inverse of the frequency response
    # - correction (Hinv is nondimensional so can apply directly to Pk_gradT)
    return Hinv


def get_noise_empirical(
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
