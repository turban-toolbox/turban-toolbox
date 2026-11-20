import json
from functools import wraps
from typing import cast, Literal
import numpy as np
from numpy import ndarray, newaxis
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft, ifft, fftfreq
from jaxtyping import Float, Int, Num, Bool, Shaped
from netCDF4 import Dataset
import xarray as xr

from turban.utils.logging import get_logger

logger = get_logger(__name__)


def kolmogorov_length(
    eps: Float[ndarray, "*any"],
    molvisc: Float[ndarray, "*any"],
) -> Float[ndarray, "*any"]:
    """Compute the Kolmogorov length scale.

    Parameters
    ----------
    eps : Float[ndarray, "*any"]
        TKE dissipation rate in W/kg.
    molvisc : Float[ndarray, "*any"]
        Kinematic viscosity in m^2/s.

    Returns
    -------
    Float[ndarray, "*any"]
        Kolmogorov length scale in m.
    """
    return (molvisc**3 / eps) ** 0.25


def average_waveno_spectra(
    psi: Float[ndarray, "*any time nfreq"],
    waveno: Float[ndarray, "time nfreq"],
    bins: Float[ndarray, "nedge"] | None = None,
) -> tuple[Float[ndarray, "*any nbin"], Float[ndarray, "nbin"]]:
    """Average wavenumber spectra over time by binning in wavenumber space.

    Each time step has its own wavenumber axis (``waveno = freq / senspeed``),
    so averaging along ``time`` at a fixed frequency index would mix different
    wavenumbers. Instead, all spectral values are sorted into common wavenumber
    bins and averaged per bin. Non-finite values are ignored.

    Parameters
    ----------
    psi : Float[ndarray, "*any time nfreq"]
        Wavenumber spectra.
    waveno : Float[ndarray, "time nfreq"]
        Wavenumber of each spectral value in ``psi``.
    bins : Float[ndarray, "nedge"], optional
        Monotonically increasing wavenumber bin edges. If None, ``nfreq`` equally
        spaced bins spanning the finite range of ``waveno`` are used.

    Returns
    -------
    psi_mean : Float[ndarray, "*any nbin"]
        Bin-averaged spectra; NaN for empty bins.
    waveno_mean : Float[ndarray, "nbin"]
        Bin centers.
    """
    *lead, ntime, nfreq = psi.shape
    if bins is None:
        bins = np.linspace(np.nanmin(waveno), np.nanmax(waveno), nfreq + 1)
    bins = cast(Float[ndarray, "nedge"], bins)
    if np.any(np.diff(bins) <= 0):
        raise ValueError("bins must be monotonically increasing")
    nbin = len(bins) - 1

    w = waveno.ravel()
    p = psi.reshape(-1, ntime * nfreq)

    idx = np.searchsorted(bins, w, side="right") - 1
    idx[w == bins[-1]] = nbin - 1  # include the right edge in the last bin
    valid = np.isfinite(w) & (idx >= 0) & (idx < nbin)
    p, idx = p[:, valid], idx[valid]

    good = np.isfinite(p)
    sums = np.zeros((p.shape[0], nbin))
    counts = np.zeros((p.shape[0], nbin))
    np.add.at(sums, (slice(None), idx), np.where(good, p, 0.0))
    np.add.at(counts, (slice(None), idx), good)
    with np.errstate(invalid="ignore", divide="ignore"):
        psi_mean = np.where(counts > 0, sums / counts, np.nan)

    return psi_mean.reshape(*lead, nbin), 0.5 * (bins[1:] + bins[:-1])


def time_mean_waveno_xarray(ds: xr.Dataset, **kwargs) -> xr.Dataset:
    """Average a dataset over time, bin-averaging spectra in wavenumber space.

    xarray wrapper around `average_waveno_spectra` for Level 3 datasets as
    produced by the ``api.py`` modules. Variables of dimension
    ``nfreq`` only (e.g. ``freq``) are broadcast along ``time``, bin-averaged and
    stored as ``<name>_mean``, replacing the original. Other variables on
    ``time`` are averaged plainly.

    Parameters
    ----------
    ds : xr.Dataset
        Level 3 dataset with a 2-D coordinate ``waveno(time, nfreq)``.
    **kwargs
        Forwarded to `average_waveno_spectra` (e.g. ``bins``).

    Returns
    -------
    xr.Dataset
        Dataset averaged over ``time`` with a 1-D ``waveno(nfreq)`` coordinate.
    """
    waveno = ds["waveno"]
    k = waveno.values
    binned = [v for v in ds.variables if "nfreq" in ds[v].dims and v != "waveno"]
    out = ds.drop_vars([*binned, "waveno"]).mean("time")

    waveno_mean = None
    for v in binned:
        da = ds[v]
        arr = da.values
        if "time" in da.dims:
            name = v
        else:
            name = f"{v}_mean"
            arr = np.broadcast_to(
                arr[..., newaxis, :], (*arr.shape[:-1], len(k), arr.shape[-1])
            )
        arr_mean, waveno_mean = average_waveno_spectra(arr, k, **kwargs)
        dims = tuple(d for d in da.dims if d != "time")
        new = xr.DataArray(arr_mean, dims=dims, attrs=da.attrs)
        if v in ds.coords:
            out = out.assign_coords({name: new})
        else:
            out[name] = new
    if waveno_mean is not None:
        out = out.assign_coords(waveno=("nfreq", waveno_mean, waveno.attrs))
    return out


def ensure_reshape_index(func):
    """Decorate `func` to accept alternative chunking parameters.

    Enables `func` to accept `section_number_or_data_len`, `segment_length`,
    `segment_overlap`, `chunk_length`, and `chunk_overlap` as alternatives to
    `reshape_index`.

    Parameters
    ----------
    func : callable
        Function to decorate.

    Returns
    -------
    callable
        Decorated function with reshape_index parameter resolution.
    """

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
    """Estimate sinking velocity from raw pressure by low-pass filtering and differentiation.

    Parameters
    ----------
    pressure_raw : array_like
        Raw pressure time series.
    sampfreq : float, optional
        Sampling frequency in Hz. Default is 1024.0.

    Returns
    -------
    vsink : ndarray
        Estimated sinking velocity (dP/dt of low-pass filtered pressure).
    pressure_lp : ndarray
        Low-pass filtered pressure time series.
    """
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


def agg_fast_to_slow(
    x: Shaped[ndarray, "*any time_fast"],
    section_number_or_data_len: Int[ndarray, "num_data"] | int | None = None,
    chunk_length: int | None = None,
    chunk_overlap: int | None = None,
    agg_method: Literal["take_first", "take_mid", "take_last"] | str = "mean",
    reshape_index: Int[ndarray, "diss_chunk fft_chunk"] | None = None,
) -> Shaped[ndarray, "*any time_slow"]:
    """
    Aggregate any quantities from fast sampling rate (e.g., shear timeseries)
    to slow sampling rate (e.g, spectra).
    If reshape_index is supplied, `section_number_or_data_len`, `chunk_length` and
    `chunk_overlap` are disregarded

    Parameters
    ----------
    method:
        "take_first": use first value of every chunk
        "take_mid": use midpoint value of every chunk
        "take_last": use last value of every chunk
        any other: use numpy function (e.g., "mean", "max", ... )

    `agg_method` can be anything that is an attribute of a numpy array, e.g. `mean`,
    `max`, etc.
    """
    if reshape_index is None:
        ii = get_chunking_index(
            section_number_or_data_len,
            (chunk_length, chunk_overlap),
        )
    else:
        ii = reshape_index

    match agg_method:
        case "take_first":
            return x[..., ii[:, 0]]
        case "take_mid":
            return x[..., ii[:, ii.shape[1] // 2]]
        case "take_last":
            return x[..., ii[:, -1]]
        case "grad":
            raise NotImplementedError("Cannot calculate gradients yet")

    # no other method fit the bill, so look in numpy:
    xi: Shaped[ndarray, "*any time_slow chunk_length"] = x[..., ii]
    return getattr(np, agg_method)(xi, axis=-1)


def agg_fast_to_slow_batch(
    data: dict[str, Num[ndarray, "time_fast"]],
    *argv,
    **kwarg,
):
    arr_fast = np.stack((arr for _, arr in data.items()), axis=0)
    arr_slow = agg_fast_to_slow(arr_fast, *argv, **kwarg)
    data_slow = {name: arr_slow[ind, :] for ind, name in enumerate(data.keys())}
    return data_slow


def get_chunking_index(
    section_number_or_data_len: Int[ndarray, "time_fast"] | int,
    *length_and_overlap: tuple[int, int],
) -> Int[ndarray, "*any"]:
    """Create index array for rechunking a dimension into overlapping segments.

    Parameters
    ----------
    section_number_or_data_len : Int[ndarray, "time_fast"] or int
        Section markers (array of int) or data length (int).
    *length_and_overlap : tuple[int, int]
        Variable number of (length, overlap) tuples specifying chunk parameters
        to be applied successively.

    Returns
    -------
    Int[ndarray, "*any"]
        Index array for chunking.
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
    """Split array into segments based on section markers.

    Parameters
    ----------
    data : Num[ndarray, "... time"]
        Data array to split.
    section_numbers : Int[ndarray, "... time"]
        Marker array specifying section membership. Zero markers are excluded.

    Returns
    -------
    dict[np.int_ | int, Num[ndarray, "... time"]]
        Dictionary mapping section marker to corresponding data segment.
        Marker 0 is not included.
    """

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
    ensure_not_nan: bool = False,
) -> Float[ndarray, "... time"]:
    """Compute gradient via FFT along last axis.

    Parameters
    ----------
    signal : Float[ndarray, "... time"]
        Input signal.
    dt : float
        Time step in seconds.
    ensure_not_nan : bool, optional
        If True, NaNs in ``signal`` are filled before the FFT (which would
        otherwise turn the whole output NaN), by linear interpolation along the
        last axis and nearest-neighbour extrapolation at the ends. ``signal``
        itself is not modified. Default is False.

    Returns
    -------
    Float[ndarray, "... time"]
        Gradient computed via FFT.
    """
    if ensure_not_nan and np.isnan(signal).any():
        signal = np.array(signal, dtype=float)  # copy, leave input untouched
        rows = signal.reshape(-1, signal.shape[-1])  # view into the copy
        idx = np.arange(signal.shape[-1])
        nnan = 0
        for row in rows:
            isnan = np.isnan(row)
            if isnan.any() and not isnan.all():
                # np.interp holds the first/last valid value outside the data range
                row[isnan] = np.interp(idx[isnan], idx[~isnan], row[~isnan])
                nnan += isnan.sum()
        logger.warning(
            f"fft_grad: filled {nnan} NaN(s) by linear interpolation"
        )
    N = signal.shape[-1]
    x = np.concatenate(
        (signal, signal[..., ::-1]), axis=-1
    )  # make periodic to avoid spectral leakage
    f = fftfreq(2 * N, dt)
    f_broadcast = atleast_nd_last(f, x.shape)
    dxdt = ifft(2 * np.pi * f_broadcast * 1j * fft(x, axis=-1), axis=-1)
    return dxdt[..., :N].real


def atleast_nd_last(arr: Float[ndarray, "... dim0"], targetshape: tuple[int, ...]):
    """Prepend size-1 axes until ``arr`` has the same number of dimensions as ``targetshape``.

    Parameters
    ----------
    arr : ndarray, shape (..., dim0)
        Input array to expand.
    targetshape : tuple of int
        Target shape whose number of dimensions is the goal.

    Returns
    -------
    ndarray
        Array with leading size-1 axes added as needed.
    """
    for _ in range(len(targetshape) - len(arr.shape)):
        arr = arr[np.newaxis, ...]
    return arr


def butterfilt(signal, cutoff_freq_Hz, sampfreq, axis=-1, **kwarg):
    """Apply first-order Butterworth filter.

    Parameters
    ----------
    signal : array_like
        Input signal.
    cutoff_freq_Hz : float
        Cutoff frequency in Hz.
    sampfreq : float
        Sampling frequency in Hz.
    axis : int, optional
        Axis along which to filter. Default is -1.
    **kwarg
        Additional keyword arguments passed to `scipy.signal.butter`.

    Returns
    -------
    ndarray
        Filtered signal.
    """
    # nondimensionalize using Nyquist freq
    cutoff_nondim = cutoff_freq_Hz / (sampfreq / 2)
    b, a = butter(N=1, Wn=cutoff_nondim, **kwarg)
    return filtfilt(b, a, signal, axis=axis)


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
    """Create index array for overlapping window segmentation.

    Parameters
    ----------
    w : int
        Window length (positive integer).
    overlap : int
        Overlap length (positive integer), must be less than `w`.
    N : int
        Dimension length to expand (positive integer).

    Returns
    -------
    Int[ndarray, "segment w"]
        Index array of shape (segment, w) for overlapping windows.
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
    """Reshape first dimension into overlapping segments.

    Parameters
    ----------
    P : Float[ndarray, "samples ..."]
        Input array.
    chunklen : int
        Chunk length.
    chunkoverlap : int
        Chunk overlap length.

    Returns
    -------
    Float[ndarray, "segment inside ..."]
        Reshaped array, or zero array if dimension is smaller than `chunklen`.
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
    """Reshape next-to-last dimension into overlapping segments.

    Parameters
    ----------
    P : Float[ndarray, "... samples _"]
        Input array.
    chunklen : int
        Chunk length.
    chunkoverlap : int
        Chunk overlap length.

    Returns
    -------
    Float[ndarray, "... segment inside _"]
        Reshaped array, or zero array if dimension is smaller than `chunklen`.
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
    """Reshape last dimension into overlapping segments.

    Parameters
    ----------
    P : Float[ndarray, "... samples"]
        Input array.
    chunklen : int
        Chunk length.
    chunkoverlap : int
        Chunk overlap length.

    Returns
    -------
    Float[ndarray, "... segment inside"]
        Reshaped array, or zero array if dimension is smaller than `chunklen`.
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
    """Reshape first dimension into half-overlapping windows.

    Parameters
    ----------
    y : Float[ndarray, "samples ..."]
        Input array.
    w : int
        Window length (even integer).

    Returns
    -------
    Float[ndarray, "segment inside ... "]
        Reshaped array with half-overlapping windows.
    """
    assert w % 2 == 0  # function would work for uneven w but results may be unintuitive
    return y[reshape_overlap_index(w, w // 2, y.shape[0]), ...]


def reshape_halfoverlap_last(
    y: Float[ndarray, "... samples"], w: int
) -> Float[ndarray, "... segment w"]:
    """Reshape last dimension into half-overlapping windows.

    Parameters
    ----------
    y : Float[ndarray, "... samples"]
        Input array.
    w : int
        Window length (even integer).

    Returns
    -------
    Float[ndarray, "... segment w"]
        Reshaped array with half-overlapping windows.
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
    return np.trapezoid(y_zero, x=x)


def integrate(
    y: Float[ndarray, "... time frequency"],
    x: Float[ndarray, "... time frequency"],
    x_from: Float[ndarray, "... time"],
    x_to: Float[ndarray, "... time"],
) -> Float[ndarray, "... time"]:
    """Integrate `y` over `x` along last axis using trapezoidal rule.

    Parameters
    ----------
    y : Float[ndarray, "... time frequency"]
        Integrand values.
    x : Float[ndarray, "... time frequency"]
        Corresponding x-axis values.
    x_from : Float[ndarray, "... time"]
        Lower integration limits.
    x_to : Float[ndarray, "... time"]
        Upper integration limits.

    Returns
    -------
    Float[ndarray, "... time"]
        Integral along last axis.
    """
    y_zero = np.where((x_from[..., newaxis] <= x) & (x <= x_to[..., newaxis]), y, 0.0)
    # TODO: handle all-nan spectra
    return np.trapezoid(y_zero, x=x, axis=-1)


def diss_chunk_wise_reshape_index(
    reshape_index: Int[ndarray, "diss_chunk fft_chunk segment_length"],
) -> Int[ndarray, "diss_chunk chunk_length"]:
    """Flatten last two dimensions while ensuring unique indices per diss_chunk.

    Parameters
    ----------
    reshape_index : Int[ndarray, "diss_chunk fft_chunk segment_length"]
        3D index array to flatten.

    Returns
    -------
    Int[ndarray, "diss_chunk chunk_length"]
        2D index array with unique indices for each diss_chunk.
    """
    ii = reshape_index
    ii_flat = ii.reshape((ii.shape[0], ii.shape[1] * ii.shape[2]))
    chunk_length = ii_flat[0].max() - ii_flat[0].min() + 1
    return ii_flat.min(axis=1)[:, newaxis] + np.arange(0, chunk_length)[newaxis, :]


def boolarr_to_sections(bools: Bool[ndarray, "time"]) -> list[list[int]]:
    """Partition boolean array into contiguous True-valued index ranges.

    Parameters
    ----------
    bools : Bool[ndarray, "time"]
        Boolean array.

    Returns
    -------
    list[list[int]]
        List of contiguous index ranges where `bools` is True.
    """
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
    segment_min_len: int = 0,
    trim: int = 0,
) -> Int[ndarray, "*any"]:
    """Identify contiguous sections where all data fall within specified bounds.

    Parameters
    ----------
    *data_and_bounds : tuple[Float[ndarray, "*any"], float, float]
        Variable number of (data_array, lo, up) tuples specifying data arrays
        and their lower and upper bounds. Sections must satisfy all bounds.
    segment_min_len : int, optional
        Minimum segment length to retain. Default is 0.
    trim : int, optional
        Positive values shrink sections by this amount; negative values widen them.
        Default is 0.

    Returns
    -------
    Int[ndarray, "*any"]
        Section numbers (0 for invalid regions, positive integers for valid sections).
    """
    data_shp = data_and_bounds[0][0].shape  # select first data entry as sample
    inds = np.ones(data_shp, dtype=bool)
    for data, lo, up in data_and_bounds:
        if lo is not None:
            inds = inds & (lo <= np.array(data))
        if up is not None:
            inds = inds & (up >= np.array(data))

    if trim != 0:
        abstrim = abs(trim)
        inds_rollpos = np.roll(inds, abstrim)
        inds_rollneg = np.roll(inds, -abstrim)
        # handle wraparound of np.roll
        inds_rollpos[:abstrim] = False
        inds_rollneg[-abstrim:] = False

        if trim > 0:
            # trim so that sections become shorter
            inds = inds & inds_rollneg & inds_rollpos
        else:
            # sections become longer
            inds = inds | inds_rollneg | inds_rollpos

    sections = boolarr_to_sections(inds)

    if segment_min_len > 0:
        sections = [sec for sec in sections if len(sec) >= segment_min_len]

    section_number = np.zeros(data_shp, dtype=int)
    for k, sec in enumerate(sections):
        section_number[sec] = k + 1

    return section_number


def unwrap_base2(
    q: Int[ndarray, "*any"],
    maxq: int | None = None,
) -> dict[int, Bool[ndarray, "*any"]]:
    """Decompose integers into base-2 bit flags.

    Parameters
    ----------
    q : Int[ndarray, "*any"]
        Integer array to decompose.
    maxq : int, optional
        Maximum value to determine bit depth. If None, uses `nanmax(q)`.

    Returns
    -------
    dict[int, Bool[ndarray, "*any"]]
        Dictionary mapping power-of-2 integers to boolean arrays indicating
        which bits are set in corresponding `q` elements.
    """
    if maxq is None:
        maxq = np.nanmax(q)

    flag_arr: Bool[ndarray, "*any"] = np.unpackbits(
        q.astype(np.uint8)[np.newaxis], axis=0, bitorder="little"
    ).astype(bool)
    base = [2**i for i in range(int(np.log2(maxq) + 1))]
    flag_dict = {name: val for name, val in zip(base, flag_arr)}
    return flag_dict
