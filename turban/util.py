import json
import numpy as np
from numpy import ndarray, newaxis
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft, ifft, fftfreq
from jaxtyping import Float, Int
from beartype.typing import Tuple


def is_valid_turban_netcdf(fname: str):
    raise NotImplementedError


def convert_atomix_benchmark_to_turban_netcdf(fname: str):
    raise NotImplementedError


def fft_grad(
    signal: Float[ndarray, "... time"],
    dt: float,
) -> Float[ndarray, "... freq"]:
    """compute gradient using FFT."""
    N = signal.shape[-1]
    x = np.concatenate(
        (signal, signal[::-1]), axis=-1
    )  # make periodic to avoid spectral leakage
    f = fftfreq(2 * N, dt)
    f_broadcast = atleast_nd_last(f, x.shape)
    dxdt = ifft(2 * np.pi * f_broadcast * 1j * fft(x, axis=-1), axis=-1)
    return dxdt[..., :N].real


def atleast_nd_last(arr: Float[ndarray, "... dim0"], targetshape: Tuple[int, ...]):
    for _ in range(len(targetshape) - len(arr.shape)):
        arr = arr[np.newaxis, ...]
    return arr


def butterfilt(signal, cutoff_freq_Hz, sampling_freq, **kwarg):
    """Apply first oder Butterworth filter. kwarg are passed into `butter`"""
    # nondimensionalize using Nyquist freq
    cutoff_nondim = cutoff_freq_Hz / (sampling_freq / 2)
    b, a = butter(N=1, Wn=cutoff_nondim, **kwarg)
    return filtfilt(b, a, signal)


def channel_mapping(json_fname, *channel_names):
    with open(json_fname, "r") as f:
        cfg = json.load(f)
    channels = [
        attrs["channel"]
        for name in channel_names
        for attrs in cfg["sensors"].values()
        if attrs["name"] == name
    ]
    return channels


def reshape_overlap_index(w: int, overlap: int, N: int) -> Int[ndarray, "segment w"]:
    """
    Expand dimension into two dimensions of overlapping intervals.
    Returns the index that expands a dimension of length N.

    N: length of dimension to be expanded (positive integer)
    w: window length (positive integer)
    overlap: overlap length (positive integer)
    """
    assert w > overlap
    # assert N >= w
    stepsize = w - overlap  # increase for each window
    row_to_row_offset = np.arange(0, N - w + 1, stepsize)  # start of each window
    nrows = len(row_to_row_offset)
    ii = (
        np.zeros((nrows, w)) + np.arange(w)[newaxis, :] + row_to_row_offset[:, newaxis]
    ).astype(int)
    return ii


def reshape_any_first(
    P: Float[ndarray, "samples ..."],
    chunklen: int,
    chunkoverlap: int,
) -> Float[ndarray, "segment inside ..."]:
    """
    Re-arrange first dimension of P
    """
    n = P.shape[0]
    if n >= chunklen:
        ii = reshape_overlap_index(chunklen, chunkoverlap, n)
        return P[np.array(ii), ...]
    else:
        return np.zeros((0, chunklen) + P.shape[1:], dtype=float)


def reshape_any_nextlast(
    P: Float[ndarray, "... samples _"],
    chunklen: int,
    chunkoverlap: int,
) -> Float[ndarray, "... segment inside _"]:
    """
    Re-arrange next-to-last dimension of P
    """
    n = P.shape[-2]
    if n >= chunklen:
        ii = reshape_overlap_index(chunklen, chunkoverlap, n)
        return P[..., np.array(ii), :]
    else:
        return np.zeros(P.shape[:-2] + (0, chunklen, P.shape[-1]), dtype=float)


def reshape_any_last(
    P: Float[ndarray, "... samples"],
    chunklen: int,
    chunkoverlap: int,
) -> Float[ndarray, "... segment inside"]:
    """
    Re-arrange last dimension of P
    """
    n = P.shape[-1]
    if n >= chunklen:
        ii = reshape_overlap_index(chunklen, chunkoverlap, n)
        return P[..., np.array(ii)]
    else:
        return np.zeros(P.shape[:-1] + (0, chunklen), dtype=float)


def reshape_halfoverlap_first(
    y: Float[ndarray, "samples ..."], w: int
) -> Float[ndarray, "segment inside ... "]:
    """
    Expand the first dimension into two dimensions of half-overlapping intervals
    w: window length (even integer)
    """
    assert w % 2 == 0  # function would work for uneven w but results may be unintuitive
    return y[reshape_overlap_index(w, w // 2, y.shape[0]), ...]


def reshape_halfoverlap_last(
    y: Float[ndarray, "... samples"], w: int
) -> Float[ndarray, "... segment w"]:
    """
    Expand the last dimension into two dimensions of half-overlapping intervals
    w: window length (even integer)
    """
    assert w % 2 == 0  # function would work for uneven w but results may be unintuitive
    return y[..., reshape_overlap_index(w, w // 2, y.shape[-1])]


def average_fast_to_slow(
    x: Float[ndarray, "*any time_fast"],
    window: int,
    chunklen: int,
    chunkoverlap: int,
) -> Float[ndarray, "*any time_slow"]:
    """
    TODO: docstring
    """
    # average in half-overlapping windows
    halfoverlapping = reshape_halfoverlap_last(x, window).mean(axis=-1)
    # average in chunks
    x_slow = reshape_any_last(
        halfoverlapping,
        chunklen,
        chunkoverlap,
    ).mean(axis=-1)
    return x_slow


def _integrate_simple(
    y: ndarray,
    x: ndarray,
    x_from: float,
    x_to: float,
):
    y_zero = np.where((x_from <= x) & (x <= x_to), y, 0)
    return np.trapz(y_zero, x=x)


def integrate(
    y: Float[ndarray, "... time frequency"],
    x: Float[ndarray, "... time frequency"],
    x_from: Float[ndarray, "... time"],
    x_to: Float[ndarray, "... time"],
) -> Float[ndarray, "... time"]:
    """
    Integrate along last axis
    """
    y_zero = np.where((x_from[..., newaxis] <= x) & (x <= x_to[..., newaxis]), y, 0.0)
    # TODO: handle all-nan spectra
    return np.trapz(y_zero, x=x, axis=-1)


def binned_gradient_halfoverlap(
    x: Float[ndarray, "time"],  # time series
    platform_speed: Float[ndarray, "time"],
    chunklen_samples: int,
    sampling_frequency: float,  # samples/second
) -> Float[ndarray, "time_agg"]:
    """
    Calculate spatial gradient in half-overlapping intervals
    """
    x = reshape_halfoverlap_last(x, chunklen_samples)
    pspda = reshape_halfoverlap_last(platform_speed, chunklen_samples).mean(axis=1)
    pspda
    # dummy time vector in seconds
    time = np.linspace(1, chunklen_samples / sampling_frequency, chunklen_samples)
    time, x.transpose()
    dxdt = np.polyfit(x=time, y=x.transpose(), deg=1)[0, :]
    dxdt
    dxdz = dxdt / pspda
    return dxdz
