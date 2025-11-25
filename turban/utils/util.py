import json
from functools import wraps
from typing import cast
import warnings
import numpy as np
from jaxtyping import Bool
from numpy import ndarray, newaxis
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft, ifft, fftfreq
from jaxtyping import Float, Int, Num
from netCDF4 import Dataset
import xarray as xr


def ensure_reshape_index(func):
    """Make sure that `func` has `reshape_index` available by alternatively supplying
    segment_length etc."""

    @wraps(func)
    def decorated(
        *argv,
        section_number_or_data_len: Int[ndarray, "num_data"] | int | None = None,
        segment_length: int | None = None,
        segment_overlap: int | None = None,
        chunk_length: int | None = None,
        chunk_overlap: int | None = None,
        **kwarg,
    ):
        if "reshape_index" not in kwarg or kwarg["reshape_index"] is None:
            # now the other parameters cannot be None
            section_number_or_data_len = cast(
                Int[ndarray, "num_data"] | int, section_number_or_data_len
            )
            segment_length = cast(int, segment_length)
            segment_overlap = cast(int, segment_overlap)
            chunk_length = cast(int, chunk_length)
            chunk_overlap = cast(int, chunk_overlap)
            kwarg["reshape_index"] = get_chunking_index(
                section_number_or_data_len,
                (chunk_length, chunk_overlap),
                (segment_length, segment_overlap),
            )
        elif (
            segment_length is not None
            or segment_overlap is not None
            or chunk_length is not None
            or chunk_overlap is not None
            or section_number_or_data_len is not None
        ):
            raise Warning(
                (
                    "Disregarding superfluous parameters: ",
                    "section_number_or_data_len, ",
                    "segment_length, ",
                    "segment_overlap, ",
                    "chunk_length, ",
                    "chunk_overlap.",
                )
            )

        return func(*argv, **kwarg)

    return decorated


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


def get_vsink(pressure_raw, sampfreq=1024.0):
    # lowpass filter pressure
    pressure_lp = butterfilt(
        signal=pressure_raw,
        cutoff_freq_Hz=0.5,
        sampfreq=sampfreq,
        btype="low",
    )
    # sinking speed
    vsink = fft_grad(pressure_lp, 1 / sampfreq)
    return vsink, pressure_lp


def fast_to_slow_grad_by_segment(
    x: Float[ndarray, "... time_fast"],
    y: Float[ndarray, "... time_fast"],
    sampfreq: float,
    segment_length: int = None,
    segment_overlap: int = None,
    chunk_length: int = None,
    chunk_overlap: int = None,
    section_number: Int[ndarray, "time_fast"] = None,
    reshape_index: Int[ndarray, "diss_chunk fft_chunk segment_length"] = None,
) -> Float[ndarray, "... time_slow"]:
    """
    Calculate the gradient of `y` with respect to `x`, averaged over each segment.
    If reshape_index is not supplied, calculates it.
    """
    if reshape_index is None:
        reshape_index = get_chunking_index(
            section_number,
            (chunk_length, chunk_overlap),
            (segment_length, segment_overlap),
        )
    else:
        segment_length = reshape_index.shape[-1]

    x = x[..., reshape_index]
    y = y[..., reshape_index]

    # dummy time vector in seconds
    time = np.linspace(1, segment_length / sampfreq, segment_length)
    dxdt = np.polyfit(x=time, y=x.transpose(), deg=1)[0, :]
    dydt = np.polyfit(x=time, y=y.transpose(), deg=1)[0, :]
    dydx = dydt / dxdt
    return dydx.mean(axis=-1)  # average gradient over each `chunk_length` segment


@ensure_reshape_index
def agg_fast_to_slow(
    x: Num[ndarray, "*any time_fast"],
    reshape_index: Int[ndarray, "diss_chunk fft_chunk segment_length"],
    agg_method: str = "mean",
) -> Num[ndarray, "*any time_slow"]:
    """
    Aggregate any quantities from fast sampling rate (e.g., shear timeseries)
    to slow sampling rate (e.g, spectra).

    `agg_method` can be anything that is an attribute of a numpy array, e.g. `mean`,
    `max`, etc.
    """
    if agg_method in ("grad"):
        raise NotImplementedError("Cannot calculate gradients yet")
    ii = diss_chunk_wise_reshape_index(reshape_index)
    xi = x[..., ii]
    return getattr(xi, agg_method)(axis=-1)
    # return getattr(np, agg_method)(xi, axis=-1) # TODO implement


def agg_fast_to_slow_batch(
    data: dict[str, Num[ndarray, "time_fast"]],
    *argv,
    **kwarg,
):
    arr_fast = np.stack((arr for _, arr in data.items()), axis=0)
    arr_slow = agg_fast_to_slow(arr_fast, *argv, **kwarg)
    data_slow = {name: arr_slow[ind, :] for ind, name in enumerate(data.keys())}
    return data_slow


def fast_to_slow_avg_by_segment():
    pass


def get_chunking_index(
    section_number_or_data_len: Int[ndarray, "time_fast"] | int,
    *length_and_overlap: tuple[int, int],
) -> Int[ndarray, "*any"]:
    """
    Create index that rechunks a dimension into overlapping segments.

    First argument: Either int (length of data) or a 1d array of int section markers.
    Successive arguments: Tuples of the form (length, overlap). These will be successively nested.
    """

    if isinstance(section_number_or_data_len, int):
        data_len = section_number_or_data_len
        section_number = np.ones(data_len, dtype=int)
    else:
        section_number = section_number_or_data_len
        data_len = len(section_number_or_data_len)

    indices = np.arange(data_len)
    sections = split_data(indices, section_number)

    for k, (length, overlap) in enumerate(length_and_overlap):
        if k == 0:
            reshape_segments = []
            for section in sections.values():
                ii: Int[ndarray, "Nchunks chunklen"] = reshape_overlap_index(
                    length, overlap, len(section)
                )
                reshape_segments.append(ii + section[0])
                ichunk = np.concatenate(reshape_segments, axis=0)

        else:
            # successive iterations
            ii: Int[ndarray, "Nchunks chunklen"] = reshape_overlap_index(
                length, overlap, ii.shape[-1]
            )
            ichunk = ichunk[..., ii]

    return ichunk


def split_data(
    data: Num[ndarray, "... time"],
    section_numbers: Int[ndarray, "... time"],
) -> dict[np.int_ | int, Num[ndarray, "... time"]]:  # sections
    """Split array of data into segments based on section markers.
    section marker "0" is neglected and not included in the output."""

    markers = set(section_numbers)
    # sections = select_sections(data_and_bounds)
    sections = {
        marker: data[..., section_numbers == marker]
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


def butterfilt(signal, cutoff_freq_Hz, sampfreq, **kwarg):
    """Apply first oder Butterworth filter. kwarg are passed into `butter`"""
    # nondimensionalize using Nyquist freq
    cutoff_nondim = cutoff_freq_Hz / (sampfreq / 2)
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


@ensure_reshape_index
def get_cleaned_fraction(
    x: Float[ndarray, "*any time_fast"],
    x_clean: Float[ndarray, "*any time_fast"],
    reshape_index: Int[ndarray, "diss_chunk fft_chunk segment_length"] | None = None,
) -> Float[ndarray, "*any time_slow"]:
    is_cleaned = x != x_clean
    ii = diss_chunk_wise_reshape_index(reshape_index)

    num_cleaned_samples = is_cleaned[..., ii].sum(axis=-1)
    num_total_samples = ii.shape[-1]
    return num_cleaned_samples / num_total_samples


def diss_chunk_wise_reshape_index(
    reshape_index: Int[ndarray, "diss_chunk fft_chunk segment_length"],
) -> Int[ndarray, "diss_chunk chunk_length"]:
    """Flatten the last two dimensions into one, making sure only unique indices appear
    for each `diss_chunk`.
    """
    ii = reshape_index
    ii_flat = ii.reshape((ii.shape[0], ii.shape[1] * ii.shape[2]))
    chunk_length = ii_flat[0].max() - ii_flat[0].min() + 1
    return ii_flat.min(axis=1)[:, newaxis] + np.arange(0, chunk_length)[newaxis, :]


def boolarr_to_sections(bools: Bool[ndarray, "time"]) -> list[list[int]]:
    """Separate a list of bools into contiguous chunks"""
    bools_as_ints = list(map(int, bools))
    sections = []
    # make sure first and last sections are picked up
    # even if bordering on list start or end
    offset = 0
    if bools[0]:
        bools_as_ints.insert(0, 0)
        offset = 1
    if bools[-1]:
        bools_as_ints.append(0)

    # register every jump from True to False
    true_to_false = np.diff(bools_as_ints) == -1
    # vice versa
    false_to_true = np.diff(bools_as_ints) == 1

    for ic0, ic1 in zip(np.flatnonzero(false_to_true), np.flatnonzero(true_to_false)):
        sections.append(list(range(ic0 + 1 - offset, ic1 + 1 - offset)))

    return sections


def define_sections(
    *data_and_bounds: tuple[Float[ndarray, "*any"], float, float],
    segment_min_len: int | None = None,
) -> Int[ndarray, "*any"]:
    """
    Select sections from a list of time series.
    Arguments are tuples of (data_array, min, max) values.
    segment_min_len (int) only retains sections with this minimum length.

    Returns:
        List of indices for each section (list of lists of integers)
    """
    data_shp = data_and_bounds[0][0].shape  # select first data entry as sample
    inds = np.ones(data_shp, dtype=bool)
    for data, lo, up in data_and_bounds:
        if lo is not None:
            inds = inds & (lo <= np.array(data))
        if up is not None:
            inds = inds & (up >= np.array(data))

    sections = boolarr_to_sections(inds)

    if segment_min_len is not None:
        sections = [sec for sec in sections if len(sec) >= segment_min_len]

    section_number = np.zeros(data_shp, dtype=int)
    for k, sec in enumerate(sections):
        section_number[sec] = k + 1

    return section_number
