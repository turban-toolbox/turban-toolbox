from typing import Literal, cast
from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int, Complex, Num
from scipy.signal import welch, csd, windows

from turban.utils.util import get_chunking_index


def spectrum(
    x: Float[ndarray, "... time_fast"],
    sampfreq: float,
    segment_length: int | None = None,
    segment_overlap: int | None = None,
    chunk_length: int | None = None,
    chunk_overlap: int | None = None,
    section_number: Int[ndarray, "time_fast"] | None = None,
    reshape_index: Int[ndarray, "diss_chunk fft_chunk segment_length"] | None = None,
    y: (
        Float[ndarray, "... time_fast"] | None
    ) = None,  # if not None, return cross spectral density
    **estimator_kwarg,
) -> tuple[
    Num[ndarray, "... chunk freq"],
    Float[ndarray, "freq"],  # frequencies
]:
    """
    Produce spectra from time series.
    If reshape_index is not supplied, calculates it.

    If onlyx is provided, will compute power spectral density.

    If y is provided (same shape as x), will compute cross spectral density.
    """
    if reshape_index is None:
        section_number = cast(Int[ndarray, "time_fast"], section_number)
        chunk_length = cast(int, chunk_length)
        chunk_overlap = cast(int, chunk_overlap)
        reshape_index = get_chunking_index(
            section_number,
            (chunk_length, chunk_overlap),
        )
    else:
        segment_length = reshape_index.shape[-1]

    kwarg = dict(
        axis=-1,
        fs=sampfreq,
        nperseg=segment_length,
        noverlap=segment_overlap,
        scaling="density",
    )

    xr = x[..., reshape_index]  # reshape to chunk  length windows

    if y is None:
        freq, psi = welch(xr, **kwarg, **estimator_kwarg)
    else:
        yr = y[..., reshape_index]
        freq, psi = csd(xr, yr, **kwarg, **estimator_kwarg)

    dfreq = freq[1] - freq[0]
    # by using scaling 'density' above, variance should already be approximately conserved,
    # but now we correct for PSD estimator bias manually:
    _ = apply_var_conserve(psi, dfreq, xr)

    return psi, freq


def apply_var_conserve(
    psi: Float[ndarray, "*any freq"],
    dfreq: float,
    signal_reshape: Float[ndarray, "nshear chunks freq"],
) -> Float[ndarray, "*any"]:
    """
    Scale `psi` such that its integral along last axis (`freq`) equals variance of input signal.
    """
    intpsi = psi[..., 1:].sum(axis=-1) * dfreq  # disregard first frequency
    signal_reshape = np.ascontiguousarray(
        signal_reshape
    )  # performance enhancement for np.var
    corr_factor = np.var(signal_reshape, axis=-1) / intpsi
    psi *= corr_factor[..., newaxis]
    return corr_factor
