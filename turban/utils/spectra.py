from typing import Literal
from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int, Complex

from turban.utils.util import get_chunking_index


def power_spectrum(
    x: Float[ndarray, "... time_fast"],
    sampling_freq: float,
    fft_length: int | None = None,
    fft_overlap: int | None = None,
    diss_length: int | None = None,
    diss_overlap: int | None = None,
    section_marker: Int[ndarray, "time_fast"] | None = None,
    reshape_index: Int[ndarray, "diss_chunk fft_chunk fft_length"] | None = None,
) -> tuple[
    Float[ndarray, "... chunk freq"],
    Float[ndarray, "freq"],  # frequencies
]:
    Pf, freq = cospectrum(
        x,
        None,
        sampling_freq,
        fft_length,
        fft_overlap,
        diss_length,
        diss_overlap,
        section_marker,
        reshape_index,
    )
    return Pf.real, freq


def cospectrum(
    x: Float[ndarray, "... time_fast"],
    y: Float[ndarray, "... time_fast"] | None,  # if None, return power spectrum of x
    sampling_freq: float,
    fft_length: int | None = None,
    fft_overlap: int | None = None,
    diss_length: int | None = None,
    diss_overlap: int | None = None,
    section_marker: Int[ndarray, "time_fast"] | None = None,
    reshape_index: Int[ndarray, "diss_chunk fft_chunk fft_length"] | None = None,
    window: Literal["hanning"] | None = "hanning",
) -> tuple[
    Complex[ndarray, "... chunk freq"],
    Float[ndarray, "freq"],  # frequencies
]:
    """
    Produce spectra from cleaned shear time series.
    If reshape_index is not supplied, calculates it.
    """
    if reshape_index is None:
        reshape_index = get_chunking_index(
            x.shape[-1],
            fft_length,
            fft_overlap,
            diss_length,
            diss_overlap,
            section_marker,
        )
    else:
        fft_length = reshape_index.shape[-1]

    freq = np.fft.rfftfreq(fft_length, d=1 / sampling_freq)

    xr = x[..., reshape_index]  # reshape to fft length windows
    xr -= xr.mean(axis=-1)[..., newaxis]  # subtract mean
    if window == "hanning":
        xr *= np.hanning(fft_length)[newaxis, newaxis, :]  # hanning window
    Fxr = np.fft.rfft(xr)[:, :]

    if y is None:
        Pf = Fxr.conj() * Fxr
    else:
        yr = y[..., reshape_index]  # reshape to fft length windows
        yr -= yr.mean(axis=-1)[..., newaxis]  # subtract mean
        if window == "hanning":
            yr *= np.hanning(fft_length)[newaxis, newaxis, :]  # hanning window
        Fyr = np.fft.rfft(yr)[:, :]
        Pf = Fyr.conj() * Fxr

    # average spectra by chunks (reshape the segments)
    Pf = Pf.mean(axis=-2)
    return Pf, freq
