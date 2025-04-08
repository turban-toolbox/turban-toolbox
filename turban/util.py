import json
from functools import wraps
import numpy as np
from numpy import ndarray, newaxis
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft, ifft, fftfreq
from jaxtyping import Float, Int, Num
from netCDF4 import Dataset
import xarray as xr


def load(fname):
    with Dataset(fname) as f:
        groups = list(f.groups)

    if "level0" in groups:
        ds0 = xr.load_dataset(fname, group="level0")
    else:
        ds0 = None

    if "microtemp" in groups:
        dst = xr.load_dataset(fname, group="microtemp")
    else:
        dst = None

    if "level3" in groups:
        ds3 = xr.load_dataset(fname, group="level3")
    else:
        ds3 = None

    if "level4" in groups:
        ds4 = xr.load_dataset(fname, group="level4")
    else:
        ds4 = None

    return ds0, ds3, ds4, dst


def is_valid_turban_netcdf(fname: str):
    raise NotImplementedError


def convert_atomix_benchmark_to_turban_netcdf(fname: str):
    raise NotImplementedError


def get_vsink(pressure_raw, sampling_freq=1024.0):
    # lowpass filter pressure
    pressure_lp = butterfilt(
        signal=pressure_raw,
        cutoff_freq_Hz=0.5,
        sampling_freq=sampling_freq,
        btype="low",
    )
    # sinking speed
    vsink = fft_grad(pressure_lp, 1 / sampling_freq)
    return vsink, pressure_lp


def fast_to_slow_grad_by_segment(
    x: Float[ndarray, "... time_fast"],
    y: Float[ndarray, "... time_fast"],
    sampling_freq: float,
    fft_length: int = None,
    fft_overlap: int = None,
    diss_length: int = None,
    diss_overlap: int = None,
    section_marker: Int[ndarray, "time_fast"] = None,
    reshape_index: Int[ndarray, "diss_chunk fft_chunk fft_length"] = None,
) -> Float[ndarray, "... time_slow"]:
    """
    Calculate the gradient of `y` with respect to `x`, averaged over each segment.
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

    x = x[..., reshape_index]
    y = y[..., reshape_index]

    # dummy time vector in seconds
    time = np.linspace(1, fft_length / sampling_freq, fft_length)
    dxdt = np.polyfit(x=time, y=x.transpose(), deg=1)[0, :]
    dydt = np.polyfit(x=time, y=y.transpose(), deg=1)[0, :]
    dydx = dydt / dxdt
    return dydx.mean(axis=-1)  # average gradient over each `diss_length` segment


def average_fast_to_slow(
    x: Float[ndarray, "*any time_fast"],
    fft_length: int = None,
    fft_overlap: int = None,
    diss_length: int = None,
    diss_overlap: int = None,
    section_marker: Int[ndarray, "time_fast"] = None,
    reshape_index: Int[ndarray, "diss_chunk fft_chunk fft_length"] = None,
) -> Float[ndarray, "*any time_slow"]:
    """
    Average any quantities from fast sampling rate (e.g., shear timeseries)
    to slow sampling rate (e.g, spectra).
    If reshape_index is not supplied, calculates it.
    """
    if reshape_index is None:
        reshape_index = fast_to_slow_reshape_index(
            x.shape[-1],
            fft_length,
            fft_overlap,
            diss_length,
            diss_overlap,
            section_marker,
        )

    # average out the two overlapping dimensions
    return x[..., reshape_index].mean(axis=-1).mean(axis=-1)


def fast_to_slow_avg_by_segment():
    pass


def fast_to_slow_reshape_index(
    data_len: int,
    fft_length: int,
    fft_overlap: int,
    diss_length: int,
    diss_overlap: int,
    section_marker: Int[ndarray, "time_fast"] | None = None,
) -> Int[ndarray, "diss_chunk fft_chunk fft_length"]:

    if section_marker is None:
        section_marker = np.ones(data_len, dtype=int)

    sections = split_data(np.arange(data_len), section_marker)

    reshape_segments = []
    for data in sections.values():

        # reshape time dimension into chunks of length diss_length
        ii_diss: Int[ndarray, "diss_chunk diss_length"] = reshape_overlap_index(
            diss_length, diss_overlap, len(data)
        )
        # reshape fft dimension into chunks of length fft_length
        ii_fft: Int[ndarray, "fft_chunk fft_length"] = reshape_overlap_index(
            fft_length, fft_overlap, ii_diss.shape[-1]
        )
        reshape_segments.append(ii_diss[:, ii_fft] + data[0])

    # concatenate along dissipation chunks
    return np.concatenate(reshape_segments, axis=0)


def split_data(
    data: Num[ndarray, "... time"],
    section_markers: Int[ndarray, "... time"],
) -> dict[np.int_ | int, Num[ndarray, "... time"]]:  # sections
    """Split array of data into segments based on section markers.
    section marker "0" is neglected and not included in the output."""

    markers = set(section_markers)
    # sections = select_sections(data_and_bounds)
    sections = {
        marker: data[..., section_markers == marker]
        for marker in markers
        if marker != 0
    }
    return sections


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


def atleast_nd_last(arr: Float[ndarray, "... dim0"], targetshape: tuple[int, ...]):
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


def ensure_reshape_index(func):
    """Make sure that `func` has `reshape_index` available by alternatively supplying
    fft_length etc."""

    @wraps(func)
    def decorated(
        *argv,
        data_len: int = None,  # length of data vector
        fft_length: int = None,
        fft_overlap: int = None,
        diss_length: int = None,
        diss_overlap: int = None,
        section_marker: Int[ndarray, "num_data"] = None,
        **kwarg,
    ):
        if "reshape_index" not in kwarg or kwarg["reshape_index"] is None:
            kwarg["reshape_index"] = fast_to_slow_reshape_index(
                data_len,
                fft_length,
                fft_overlap,
                diss_length,
                diss_overlap,
                section_marker,
            )
        return func(*argv, **kwarg)

    return decorated


@ensure_reshape_index
def get_cleaned_fraction(
    x: Float[ndarray, "*any time_fast"],
    x_clean: Float[ndarray, "*any time_fast"],
    reshape_index: Int[ndarray, "diss_chunk fft_chunk fft_length"] | None = None,
) -> Float[ndarray, "*any time_slow"]:
    is_cleaned = x != x_clean
    ii = diss_chunk_wise_reshape_index(reshape_index)

    num_cleaned_samples = is_cleaned[..., ii].sum(axis=-1)
    num_total_samples = ii.shape[-1]
    return num_cleaned_samples / num_total_samples


def diss_chunk_wise_reshape_index(
    reshape_index: Int[ndarray, "diss_chunk fft_chunk fft_length"],
) -> Int[ndarray, "diss_chunk diss_length"]:
    """Flatten the last two dimensions into one, making sure only unique indices appear
    for each `diss_chunk`.
    """
    ii = reshape_index
    ii_flat = ii.reshape((ii.shape[0], ii.shape[1] * ii.shape[2]))
    diss_length = ii_flat[0].max() - ii_flat[0].min() + 1
    return ii_flat.min(axis=1)[:, newaxis] + np.arange(0, diss_length)[newaxis, :]
