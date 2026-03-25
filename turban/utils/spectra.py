from typing import Literal, cast
from itertools import product

from numpy import ndarray, newaxis
import numpy as np
from jaxtyping import Float, Int, Complex, Num
from scipy.signal import welch, csd, windows

from turban.utils.util import get_chunking_index


def spectrum(
    x: Float[ndarray, "nx ... time_fast"],
    sampfreq: float,
    segment_length: int,
    segment_overlap: int,
    chunk_length: int | None = None,
    chunk_overlap: int | None = None,
    section_number: Int[ndarray, "time_fast"] | None = None,
    reshape_index: Int[ndarray, "diss_chunk chunk_length"] | None = None,
    y: (
        Float[ndarray, "ny ... time_fast"] | None
    ) = None,  # if not None, return cross spectral density
    kind: Literal["diagonal", "cross"] = "diagonal",
    **estimator_kwarg,
) -> tuple[
    Num[ndarray, "nx ... chunk freq"],
    Float[ndarray, "freq"],  # frequencies
]:
    """Compute power spectral density or cross spectral density from time series.

    Parameters
    ----------
    x : ndarray, shape (... time_fast)
        Input time series.
    sampfreq : float
        Sampling frequency in Hz.
    segment_length : int, optional
        FFT segment length in samples.
    segment_overlap : int, optional
        FFT segment overlap in samples.
    chunk_length : int, optional
        Dissipation chunk length.
    chunk_overlap : int, optional
        Dissipation chunk overlap.
    section_number : ndarray of int, shape (time_fast,), optional
        Section markers.
    reshape_index : ndarray of int, shape (diss_chunk, fft_chunk, segment_length), optional
        Precomputed reshaping index. If not provided, is calculated from other parameters.
    y : ndarray, shape (... time_fast), optional
        If provided, compute cross spectral density instead of PSD.
    **estimator_kwarg
        Additional keyword arguments passed to welch or csd.

    Returns
    -------
    psi : ndarray, shape (... chunk freq)
        Power spectral density (PSD) or cross spectral density (CSD).
    freq : ndarray, shape (freq,)
        Frequency array.
    """
    if reshape_index is None:
        section_number = cast(Int[ndarray, "time_fast"], section_number)
        chunk_length = cast(int, chunk_length)
        chunk_overlap = cast(int, chunk_overlap)
        reshape_index = get_chunking_index(
            section_number,
            (chunk_length, chunk_overlap),
        )

    kwarg = dict(
        axis=-1,
        fs=sampfreq,
        nperseg=segment_length,
        noverlap=segment_overlap,
        scaling="density",
    )

    xr = x[..., reshape_index]  # reshape to chunk length windows

    match kind:
        case "diagonal":
            if y is None:
                freq, psi = welch(xr, **kwarg, **estimator_kwarg)
            else:
                assert (
                    x.shape[0] == y.shape[0]
                ), f"For kind {kind}, first dimensions of x and y must match"
                yr = y[..., reshape_index]
                freq, psi = csd(xr, yr, **kwarg, **estimator_kwarg)

            dfreq = freq[1] - freq[0]
            # by using scaling 'density' above, variance should already be approximately conserved,
            # but now we correct for PSD estimator bias manually:
            _ = apply_var_conserve(psi, dfreq, xr)

        case "cross":
            nx = xr.shape[0]
            comb: Int[ndarray, "n_combinations 2"]
            if y is None:
                yr = xr
                ny = yr.shape[0]
            else:
                yr = y[..., reshape_index]  # reshape to chunk length windows
                ny = y.shape[0]

            comb = np.array(list(product(range(nx), range(ny))))
            psi: Num[ndarray, "nx*ny ... chunk freq"]
            freq, psi = csd(xr[comb[:, 0]], yr[comb[:, 1]], **kwarg, **estimator_kwarg)
            psi: Num[ndarray, "nx ny ... chunk freq"] = psi.reshape(
                nx, ny, *psi.shape[1:]
            )

    return psi, freq


def apply_var_conserve(
    psi: Float[ndarray, "*any freq"],
    dfreq: float,
    signal_reshape: Float[ndarray, "nshear chunks freq"],
) -> Float[ndarray, "*any"]:
    """Apply variance conservation correction to power spectral density.

    Scales `psi` in-place such that its integral along the frequency axis equals
    the variance of the input signal.

    Parameters
    ----------
    psi : ndarray, shape (*any, freq)
        Power spectral density. Modified in-place.
    dfreq : float
        Frequency resolution.
    signal_reshape : ndarray, shape (nshear, chunks, freq)
        Reshaped signal used for variance calculation.

    Returns
    -------
    ndarray, shape (*any,)
        Correction factors applied to psi.
    """
    intpsi = psi[..., 1:].sum(axis=-1) * dfreq  # disregard first frequency
    signal_reshape = np.ascontiguousarray(
        signal_reshape
    )  # performance enhancement for np.var
    corr_factor = np.var(signal_reshape, axis=-1) / intpsi
    psi *= corr_factor[..., newaxis]
    return corr_factor
