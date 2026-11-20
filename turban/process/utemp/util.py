import numpy as np
from jaxtyping import Float
from numpy import ndarray
from scipy.signal import butter, freqz


def correction_frequency_response_bilinear(
    freq: Float[ndarray, "freq"],
    sampfreq: float,
    diff_gain: float,
) -> Float[ndarray, "freq"]:
    """Correction of power spectra for the bilinear (digital) low-pass deconvolution.

    The pre-emphasis ``x + diff_gain * dx/dt`` of a thermistor channel is removed by
    a digital first-order Butterworth low-pass (bilinear transform, e.g.
    ``turban.instruments.mss.mss_utils.deconvolve_mss_ntchp``). The ideal analog
    low-pass has the power response ``1 / (1 + (2 pi f tau)**2)`` with
    ``tau = diff_gain``. The returned ratio of the analog to the digital power
    response multiplies power spectra of the deconvolved signal.

    Parameters
    ----------
    freq : ndarray, shape (freq,)
        Frequencies in Hz at which the correction is evaluated.
    sampfreq : float
        Sampling frequency in Hz.
    diff_gain : float
        Time constant of the pre-emphasis differentiator in s.

    Returns
    -------
    ndarray, shape (freq,)
        Multiplicative correction factor for power spectra.
    """
    b, a = butter(1, 1 / (2 * np.pi * diff_gain * sampfreq / 2))
    _, h_digital = freqz(b, a, worN=freq, fs=sampfreq)
    h_analog = 1 / (1 + (2 * np.pi * freq * diff_gain) ** 2)
    return h_analog / np.abs(h_digital) ** 2
