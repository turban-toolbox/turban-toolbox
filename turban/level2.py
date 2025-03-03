import numpy as np
from .util import butterfilt
from jaxtyping import Float, Num, Bool, Int
from beartype.typing import Tuple, Dict, List
from numpy import ndarray
import warnings


data_and_bounds_type = List[
    Tuple[
        Tuple[
            None | float,  # minimum
            None | float,  # maximum
        ],
        Float[ndarray, "time"],  # time series
    ]
]


def process_level2(
    shear: Float[ndarray, "n_shear time"],
    section_select_idx: List[List[int]],
    sampling_freq_Hz: float,
    fftlen: int,
) -> List[
    Tuple[
        Float[ndarray, "n_shear time"],  # despiked shear
        Bool[ndarray, "n_shear time"],  # boolean flag: is despiked?
        Int[ndarray, "n_shear"],  # number of despike iterations
    ]
]:
    out_segments = []
    segments = [shear[..., inds] for inds in section_select_idx]

    for data in segments:
        sh_clean = np.nan * np.zeros_like(data)
        flag1 = np.zeros_like(data).astype(bool)
        flag2 = -999 * np.zeros(data.shape[0], dtype=int)
        for k, sh in enumerate(data):
            sh, flag1[k, :], flag2[k] = clean_shear(
                sh,
                sampling_freq=sampling_freq_Hz,
                spike_threshold=8.0,
                max_tries=8,
                spike_replace_before=512,
                spike_replace_after=512,
                spike_include_before=10,
                spike_include_after=20,
            )

            # after removal of spikes, can high-pass filter
            # Eq. 17
            sh_clean[k, :] = butterfilt(
                signal=sh,
                cutoff_freq_Hz=0.5 / (fftlen / sampling_freq),
                sampling_freq=sampling_freq,
                btype="high",
            )
        out_segments.append((sh_clean, flag1, flag2))

    return out_segments


def select_sections(
    data_and_bounds: data_and_bounds_type,
    segment_min_len: int = None,
) -> List[List[int]]:
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
        sections = [sec for sec in sections if len(sec)>= segment_min_len]

    return sections


def boolarr_to_sections(bools: Bool[ndarray, "time"]) -> List[List[int]]:
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
    sections: List[List[int]],
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
    sh: Float[ndarray, "time"],
    sampling_freq: float,
    spike_threshold: float = 8.0,
    spike_include_before: int = 10,
    spike_include_after: int = 20,
) -> Bool[ndarray, "time"]:
    sh_hp = butterfilt(
        signal=sh,
        cutoff_freq_Hz=0.1,
        sampling_freq=sampling_freq,
        btype="high",
    )
    sh_abs = np.abs(sh_hp)
    sh_lp = butterfilt(
        signal=sh_abs,
        cutoff_freq_Hz=1,
        sampling_freq=sampling_freq,
        btype="lp",
    )
    spikes = (sh_abs / sh_lp) > spike_threshold  # boolean array
    spikes = enlarge_bool(spikes, spike_include_before, spike_include_after)
    return np.array(spikes)


def clean_shear(
    sh: Float[ndarray, "time"],
    sampling_freq: float,
    spike_threshold: float = 8.0,
    max_tries: int = 10,
    spike_replace_before: int = 512,
    spike_replace_after: int = 512,
    spike_include_before: int = 10,
    spike_include_after: int = 20,
) -> Tuple[
    Float[ndarray, "time"],  # despiked shear
    Bool[ndarray, "time"],  # boolean flag: is despiked?
    int,  # number of despike iterations
]:
    """Section 3.2.2"""
    N = len(sh)
    ctr = 0
    spikes = detect_shear_spikes(
        sh,
        sampling_freq,
        spike_threshold=spike_threshold,
        spike_include_before=spike_include_before,
        spike_include_after=spike_include_after,
    )
    ctr = np.zeros_like(sh, dtype=bool)
    while np.any(spikes) and (ctr < max_tries):
        sh_previous = sh.copy()
        spike_sections = boolarr_to_sections(spikes)
        spike_markers = sections_to_marker(spike_sections, N)
        sh = np.array(
            replace_spikes(
                sh,
                spike_markers,
                spike_replace_before=spike_replace_before,
                spike_replace_after=spike_replace_after,
            )
        )
        ctr[sh != sh_previous] += 1
        warnings.warn("replace_spikes not implemented in python", UserWarning)
        spikes = detect_shear_spikes(
            sh,
            sampling_freq,
            spike_threshold=spike_threshold,
            spike_include_before=spike_include_before,
            spike_include_after=spike_include_after,
        )

    altered = sh != sh_raw

    return sh, altered, ctr


def split_data(
    data: Num[ndarray, "... time"],

from numba import jit


@jit
def replace_spikes(sh, spike_markers, spike_replace_before, spike_replace_after):
    """In-place replacement of shear spikes according to atomix procedure"""
    for marker in np.unique(spike_markers):
        if marker == 0:
            continue
        (spike,) = np.where(spike_markers == marker)
        start = min(spike)
        stop = max(spike)
        replace_spike(sh, start, stop, spike_replace_before, spike_replace_after)


@jit
def replace_spike(
    sh,  #: Float[ndarray, "time"],
    start,
    stop,
    spike_replace_before,  #: int,
    spike_replace_after,  #: int,
):  # -> Float[ndarray, "time"]:
    context_mean_before = np.mean(sh[max(start - spike_replace_before, 0) : start])
    context_mean_after = np.mean(
        sh[stop + 1 : min(len(sh), stop + spike_replace_after + 1)]
    )

    sh[start:stop] = (context_mean_before + context_mean_after) / 2
