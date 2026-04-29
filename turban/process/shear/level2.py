import numpy as np

from turban.utils.util import butterfilt
from jaxtyping import Float, Bool, Int
from numpy import ndarray
from turban.utils.util import split_data, boolarr_to_sections


data_and_bounds_type = list[
    tuple[
        tuple[
            None | float,  # minimum
            None | float,  # maximum
        ],
        Float[ndarray, "time"],  # time series
    ]
]


def process_level2(
    shear: Float[ndarray, "nshear time"],
    section_numbers: Int[ndarray, "time"],
    sampfreq: float,
    segment_length: int,
    cutoff_freq_lp: float,
    spike_threshold: float,
    max_tries: int,
    spike_replace_before: int,
    spike_replace_after: int,
    spike_include_before: int,
    spike_include_after: int,
) -> tuple[
    Float[ndarray, "nshear time"],  # despiked shear
    Int[ndarray, "nshear time"],  # number of despike iterations
]:
    """Despike and high-pass filter shear time series section-by-section.

    Parameters
    ----------
    shear : ndarray, shape (nshear, time)
        Raw shear time series for each sensor.
    section_numbers : ndarray of int, shape (time,)
        Section marker array; zero marks invalid data.
    sampfreq : float
        Sampling frequency in Hz.
    segment_length : int
        Length of FFT segments (used to set the high-pass cutoff frequency).
    cutoff_freq_lp : float
        Low-pass cutoff frequency in Hz for spike detection envelope.
    spike_threshold : float
        Ratio threshold above which a sample is flagged as a spike.
    max_tries : int
        Maximum number of despike iterations per section.
    spike_replace_before : int
        Number of samples before a spike replaced during interpolation.
    spike_replace_after : int
        Number of samples after a spike replaced during interpolation.
    spike_include_before : int
        Number of samples before a detected spike also flagged as spike.
    spike_include_after : int
        Number of samples after a detected spike also flagged as spike.

    Returns
    -------
    sh_clean_agg : ndarray, shape (nshear, time)
        Despiked and high-pass filtered shear.
    ctr_agg : ndarray of int, shape (nshear, time)
        Number of despike iterations applied to each sample.
    """
    segments = split_data(shear, section_numbers)
    sh_clean_agg = np.nan * np.zeros_like(shear)
    ctr_agg = np.zeros_like(shear, dtype=int)

    for marker, data in segments.items():
        for k, shear in enumerate(data):
            shear, ctr = clean_shear(
                shear,
                sampfreq=sampfreq,
                spike_threshold=spike_threshold,
                max_tries=max_tries,
                spike_replace_before=spike_replace_before,
                spike_replace_after=spike_replace_after,
                spike_include_before=spike_include_before,
                spike_include_after=spike_include_after,
                cutoff_freq_lp=cutoff_freq_lp,
            )

            # after removal of spikes, can high-pass filter
            # Eq. 17
            sh_clean = butterfilt(
                signal=shear,
                cutoff_freq_Hz=0.5 / (segment_length / sampfreq),
                sampfreq=sampfreq,
                btype="high",
            )
            ctr_agg[k, section_numbers == marker] = ctr
            sh_clean_agg[k, section_numbers == marker] = sh_clean

    return sh_clean_agg, ctr_agg


def select_sections(
    data_and_bounds: data_and_bounds_type,
    segment_min_len: int = None,
) -> list[list[int]]:
    """Select contiguous sections where data falls within specified bounds.

    Filters data points that simultaneously satisfy all (min, max) constraints,
    then returns indices of contiguous True regions.

    Parameters
    ----------
    data_and_bounds : list of tuple
        List of ((min, max), data_array) pairs. Each pair specifies a lower
        and upper bound (either may be None) and corresponding time series.
        All constraints are AND'd together.
    segment_min_len : int, optional
        Minimum length required to include a section. If None, all sections
        are returned.

    Returns
    -------
    list of list of int
        Indices for each selected section.
    """
    data_sample = data_and_bounds[0][1]
    inds = np.ones(data_sample.shape, dtype=bool)
    for (lo, up), data in data_and_bounds:
        if lo is not None:
            inds = inds & (lo <= np.array(data))
        if up is not None:
            inds = inds & (up >= np.array(data))

    sections = boolarr_to_sections(inds)

    if segment_min_len is not None:
        sections = [sec for sec in sections if len(sec) >= segment_min_len]

    return sections


def sections_to_marker(
    sections: list[list[int]],
    n: int,  # length of time series
) -> Int[ndarray, "time"]:
    """Convert section indices to a marker array.

    Parameters
    ----------
    sections : list of list of int
        Indices for each section.
    n : int
        Length of output marker array (length of time series).

    Returns
    -------
    ndarray of int, shape (n,)
        Marker array where 0 indicates no section, 1 indicates first section,
        2 indicates second section, etc.
    """
    marker = np.zeros(n, dtype=int)
    for i, sec in enumerate(sections):
        marker[sec] = i + 1  # 0 is reserved for no section
    return marker


from functools import reduce


def rollpad1(x, n, pad):
    """Roll array and pad boundaries instead of wrapping.

    Parameters
    ----------
    x : array_like, shape (time,)
        Input array.
    n : int
        Number of positions to roll. Positive rolls right, negative rolls left.
    pad : scalar
        Fill value for the boundary region created by rolling.

    Returns
    -------
    ndarray, shape (time,)
        Rolled array with boundaries filled by `pad` instead of wrapped values.
    """
    xr = np.roll(x, n)
    if n < 0:
        xr[n:] = pad
    elif n > 0:
        xr[:n] = pad
    return xr


def enlarge_bool(x, before, after):
    """Extend True values in a boolean array left and right.

    Parameters
    ----------
    x : ndarray of bool
        Input boolean array.
    before : int
        Number of samples to extend leftward from each True value.
    after : int
        Number of samples to extend rightward from each True value.

    Returns
    -------
    ndarray of bool
        Boolean array with True regions expanded.
    """
    return reduce(
        lambda x, y: x | y, [rollpad1(x, i, False) for i in range(-before, after + 1)]
    )


def detect_shear_spikes(
    shear: Float[ndarray, "time"],
    sampfreq: float,
    spike_threshold: float,
    spike_include_before: int,
    spike_include_after: int,
    cutoff_freq_lp: float,
) -> Bool[ndarray, "time"]:
    sh_hp = butterfilt(
        signal=shear,
        cutoff_freq_Hz=0.1,
        sampfreq=sampfreq,
        btype="high",
    )
    sh_abs = np.abs(sh_hp)
    sh_lp = butterfilt(
        signal=sh_abs,
        cutoff_freq_Hz=cutoff_freq_lp,
        sampfreq=sampfreq,
        btype="lp",
    )
    spikes = (sh_abs / sh_lp) > spike_threshold  # boolean array
    spikes = enlarge_bool(spikes, spike_include_before, spike_include_after)
    return np.array(spikes)


def clean_shear(
    shear: Float[ndarray, "time"],
    sampfreq: float,
    spike_threshold: float,
    max_tries: int,
    spike_replace_before: int,
    spike_replace_after: int,
    spike_include_before: int,
    spike_include_after: int,
    cutoff_freq_lp: float,
) -> tuple[
    Float[ndarray, "time"],  # despiked shear
    Int[ndarray, "time"],  # number of despike iterations on each sample
]:
    """Iteratively detect and replace spikes in shear data (ATOMIX Section 3.2.2).

    Parameters
    ----------
    shear : ndarray, shape (time,)
        Shear time series (modified in-place).
    sampfreq : float
        Sampling frequency in Hz.
    spike_threshold : float
        Ratio threshold above which a sample is flagged as a spike.
    max_tries : int
        Maximum number of despike iterations.
    spike_replace_before : int
        Number of samples before a spike to use for interpolation context.
    spike_replace_after : int
        Number of samples after a spike to use for interpolation context.
    spike_include_before : int
        Number of samples before a detected spike also marked as spike.
    spike_include_after : int
        Number of samples after a detected spike also marked as spike.
    cutoff_freq_lp : float
        Low-pass cutoff frequency in Hz for spike detection envelope.

    Returns
    -------
    tuple of (ndarray, ndarray)
        Despiked shear, shape (time,).
        Iteration count per sample, shape (time,).
    """
    N = len(shear)
    ctr = 0
    spikes = detect_shear_spikes(
        shear,
        sampfreq,
        spike_threshold=spike_threshold,
        spike_include_before=spike_include_before,
        spike_include_after=spike_include_after,
        cutoff_freq_lp=cutoff_freq_lp,
    )
    ctr = np.zeros_like(shear, dtype=int)
    while np.any(spikes) and np.all(ctr <= max_tries):
        spike_sections = boolarr_to_sections(spikes)
        spike_markers = sections_to_marker(spike_sections, N)
        replace_spikes(
            shear,
            spike_markers,
            spike_replace_before=spike_replace_before,
            spike_replace_after=spike_replace_after,
        )
        ctr[spikes] += 1
        spikes = detect_shear_spikes(
            shear,
            sampfreq,
            spike_threshold=spike_threshold,
            spike_include_before=spike_include_before,
            spike_include_after=spike_include_after,
            cutoff_freq_lp=cutoff_freq_lp,
        )

    return shear, ctr


from numba import jit, float64
from numpy import isnan, nan


@jit(float64(float64[:]))
def nanmean_empty(x):
    """Compute nanmean while handling empty arrays (numba workaround).

    Workaround for https://github.com/numba/numba/issues/5502: numba's
    nanmean raises an exception on empty arrays, so this function returns
    nan instead.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Input array of float64.

    Returns
    -------
    float64
        Nanmean of x, or nan if x is empty.
    """
    if len(x) == 0:
        return np.nan
    else:
        return np.nanmean(x)


@jit(float64(float64, float64))
def nanmean_two(a, b):
    a_nnan = ~isnan(a)
    b_nnan = ~isnan(b)
    if a_nnan and b_nnan:
        return (a + b) / 2
    elif a_nnan:
        return a
    elif b_nnan:
        return b
    else:
        return nan


@jit  # ((float64[:], int32, int32, int32, int32))
def replace_spike(
    shear,  #: Float[ndarray, "time"],
    start,
    stop,
    spike_replace_before,  #: int,
    spike_replace_after,  #: int,
):
    context_mean_before = nanmean_empty(
        shear[max(start - spike_replace_before, 0) : start]
    )
    context_mean_after = nanmean_empty(
        shear[stop : min(len(shear), stop + spike_replace_after)]
    )

    shear[start:stop] = nanmean_two(context_mean_before, context_mean_after)


@jit
def replace_spikes(shear, spike_markers, spike_replace_before, spike_replace_after):
    """Replace spikes in shear via linear interpolation (ATOMIX procedure).

    Modifies shear in-place. Each spike region is replaced with the mean of
    values before and after, computed from non-nan samples in the specified
    context windows.

    Parameters
    ----------
    shear : ndarray, shape (time,)
        Shear time series (modified in-place).
    spike_markers : ndarray of int, shape (time,)
        Marker array where each unique nonzero value indicates one spike
        region to replace. 0 indicates no spike.
    spike_replace_before : int
        Number of samples before the spike region to use for context mean.
    spike_replace_after : int
        Number of samples after the spike region to use for context mean.

    Returns
    -------
    None
        Modifies shear in-place.
    """
    n = len(spike_markers)
    i = 0
    while i < n:
        if spike_markers[i] == 0:
            i += 1
            continue
        # Found the start of a spike region
        start = i
        current_marker = spike_markers[i]
        while i < n and spike_markers[i] == current_marker:
            i += 1
        stop = i  # exclusive end
        replace_spike(shear, start, stop, spike_replace_before, spike_replace_after)
