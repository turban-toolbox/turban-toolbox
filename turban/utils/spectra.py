from beartype.typing import Tuple

from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int

from turban.util import fast_to_slow_reshape_index


def power_spectrum(
    shear: Float[ndarray, "n_shear time_fast"],
    sampling_freq: float,
    fft_length: int = None,
    fft_overlap: int = None,
    diss_length: int = None,
    diss_overlap: int = None,
    section_marker: Int[ndarray, "time_fast"] | None = None,
    reshape_index: Int[ndarray, "diss_chunk fft_chunk fft_length"] | None = None,
) -> Tuple[
    Float[ndarray, "n_shear chunk freq"],
    Float[ndarray, "freq"],  # frequencies
]:
    """
    Produce spectra from cleaned shear time series.
    If reshape_index is not supplied, calculates it.
    """
    if reshape_index is None:
        reshape_index = fast_to_slow_reshape_index(
            shear.shape[-1],
            fft_length,
            fft_overlap,
            diss_length,
            diss_overlap,
            section_marker,
        )
    else:
        fft_length = reshape_index.shape[-1]

    # reshape to fft length windows
    yr = shear[..., reshape_index]
    # subtract mean
    yr -= yr.mean(axis=-1)[..., newaxis]
    # hanning window
    yr *= np.hanning(fft_length)[newaxis, newaxis, :]

    # periodograms
    freq = np.fft.rfftfreq(fft_length, d=1 / sampling_freq)
    Fyr = np.fft.rfft(yr)[:, :]
    Pf = (Fyr.conj() * Fyr).real
    # average spectra by chunks (reshape the segments)
    Pf = Pf.mean(axis=-2)
    return Pf, freq
