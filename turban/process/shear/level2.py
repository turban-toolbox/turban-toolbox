import numpy as np

from turban.utils.util import butterfilt
from jaxtyping import Float, Bool, Int
from numpy import ndarray
from turban.utils.util import split_data


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
    shear: Float[ndarray, "n_shear time"],
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
    Float[ndarray, "n_shear time"],  # despiked shear
    Int[ndarray, "n_shear time"],  # number of despike iterations
]:
    segments = split_data(shear, section_numbers)
    sh_clean_agg = np.nan * np.zeros_like(shear)
    ctr_agg = np.zeros_like(shear, dtype=int)

    for marker, data in segments.items():
        flag2 = -999 * np.zeros(data.shape[0], dtype=int)
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
    """
    Select sections from a list of time series. Arguments are alternatingly
    data and tuples of (min, max) values

    Returns:
        List of indices for each section (list of lists of integers)
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


def sections_to_marker(
    sections: list[list[int]],
    n: int,  # length of time series
) -> Int[ndarray, "time"]:
    """Convert a list of sections to a marker array.
    0 means no section, 1 means first section, etc."""
    marker = np.zeros(n, dtype=int)
    for i, sec in enumerate(sections):
        marker[sec] = i + 1  # 0 is reserved for no section
    return marker


from functools import reduce


def rollpad1(x, n, pad):
    """Roll but don't wrap around"""
    xr = np.roll(x, n)
    if n < 0:
        xr[n:] = pad
    elif n > 0:
        xr[:n] = pad
    return xr


def enlarge_bool(x, before, after):
    """Extend True values to their front and back"""
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
    """Section 3.2.2"""
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
    """Circumvent https://github.com/numba/numba/issues/5502"""
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
    """In-place replacement of shear spikes according to atomix procedure"""
    for marker in np.unique(spike_markers):
        if marker == 0:
            continue
        (spike,) = np.where(spike_markers == marker)
        start = min(spike)
        stop = max(spike) + 1
        replace_spike(shear, start, stop, spike_replace_before, spike_replace_after)
