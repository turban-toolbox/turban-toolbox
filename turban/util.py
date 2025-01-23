import json
import numpy as np
from numpy import ndarray, newaxis
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft, ifft, fftfreq
from jaxtyping import Float
from beartype.typing import Tuple


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


def butterfilt(signal, cutoff_freq_Hz, sampling_freq_Hz, **kwarg):
    """Apply first oder Butterworth filter. kwarg are passed into `butter`"""
    # nondimensionalize using Nyquist freq
    cutoff_nondim = cutoff_freq_Hz / (sampling_freq_Hz / 2)
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


def _reshape_any_inds(chunklen, chunkoverlap, n):
    """
    TODO: lose the while loop
    """
    i = 0
    ii = []
    while i + chunklen <= n:
        ii.append(list(range(i, i + chunklen)))
        i += chunklen - chunkoverlap
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
        ii = _reshape_any_inds(chunklen, chunkoverlap, n)
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
        ii = _reshape_any_inds(chunklen, chunkoverlap, n)
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
        ii = _reshape_any_inds(chunklen, chunkoverlap, n)
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
    if y.shape[0] >= w:
        assert (
            w % 2 == 0
        )  # function would work for uneven w but results may be unintuitive
        N = y.shape[0]
        w2 = w // 2  # half-window
        Nw = N // w  # number of whole segments
        # based on number of half-segments, decide whether last full segment has even or odd index
        if (N // w2) % 2 == 0:  # even
            nrows = 2 * Nw - 1
            sign = -1
        else:  # odd
            nrows = 2 * Nw
            sign = +1

        yr = np.nan * np.zeros((nrows, w) + y.shape[1:])

        yr[::2, ...] = y[: Nw * w, ...].reshape((Nw, w) + y.shape[1:])
        yr[1::2, ...] = y[w2 : Nw * w + sign * w2, ...].reshape(
            (nrows - Nw, w) + y.shape[1:]
        )
    else:
        yr = np.zeros((0, w) + y.shape[1:])
    return yr


def reshape_halfoverlap_last(
    y: Float[ndarray, "... samples"], w: int
) -> Float[ndarray, "... segment w"]:
    """
    Expand the last dimension into two dimensions of half-overlapping intervals
    w: window length (even integer)
    """
    if y.shape[-1] >= w:
        assert (
            w % 2 == 0
        )  # function would work for uneven w but results may be unintuitive
        # (alternating between overlapping unevenly)
        N = y.shape[-1]
        w2 = w // 2  # half-window
        Nw = N // w  # number of whole segments
        # based on number of half-segments, decide whether last full segment has even or odd index
        if (N // w2) % 2 == 0:  # even
            nrows = 2 * Nw - 1
            sign = -1
        else:  # odd
            nrows = 2 * Nw
            sign = +1

        yr = np.nan * np.zeros(y.shape[:-1] + (nrows, w))

        yr[..., ::2, :] = y[..., : Nw * w].reshape(y.shape[:-1] + (Nw, w))
        yr[..., 1::2, :] = y[..., w2 : Nw * w + sign * w2].reshape(
            y.shape[:-1] + (nrows - Nw, w)
        )
    else:
        yr = np.zeros(y.shape[:-1] + (0, w))
    return yr


def average_fast_to_slow(
    x: Float[ndarray, "... time_fast"],
    window: int,
    chunklen: int,
    chunkoverlap: int,
) -> Float[ndarray, "... time_slow"]:
    """
    TODO: docstring
    """
    halfoverlapping = reshape_halfoverlap_last(x, window).mean(axis=-1)
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
