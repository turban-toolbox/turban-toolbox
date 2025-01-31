import pytest
import numpy as np
from .level2 import (
    boolarr_to_sections,
    detect_shear_spikes,
    # replace_spikes,
    rollpad1,
    enlarge_bool,
)


def test_enlarge_bool():
    assert np.all(
        rollpad1([True, True, False], 1, False) == np.array([False, True, True])
    )

    assert np.all(
        enlarge_bool([False, True, False, False, False, True], 1, 0)
        == np.array([True, True, False, False, True, True])
    )


def test_boolarr_to_sections():
    bools = [True, False, False, True, True, False, True]
    result = [[0], [3, 4], [6]]
    assert boolarr_to_sections(bools) == result

    bools = [True, False, False, True, True, False]
    result = [[0], [3, 4]]
    assert boolarr_to_sections(bools) == result

    bools = [False, False, True, True, False, True]
    result = [[2, 3], [5]]
    assert boolarr_to_sections(bools) == result

    bools = [False, False, True, True, False]
    result = [[2, 3]]
    assert boolarr_to_sections(bools) == result


def test_despike():
    sh = np.ones(100)
    sh[50] = 100
    sh[[49, 51]] = 3
    result = np.ones(100)
    # sh[[49, 51]] = 1

    sampling_freq = 1024.
    spike_threshold = 8.
    spikes = detect_shear_spikes(
        sh,
        sampling_freq,
        spike_threshold,
        spike_include_before=10,
        spike_include_after=20,
    )

    spike_sections = boolarr_to_sections(spikes)
    sh_filtered = replace_spikes(
        sh,
        spike_sections,
        spike_replace_before=10,
        spike_replace_after=10,
    )

    assert np.all(result == sh_filtered)
