import numpy as np
from turban.shear.level2 import (
    boolarr_to_sections,
    sections_to_marker,
    detect_shear_spikes,
    replace_spikes,
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
    bools = np.array([True, False, False, True, True, False, True])
    result = [[0], [3, 4], [6]]
    assert boolarr_to_sections(bools) == result

    bools = np.array([True, False, False, True, True, False])
    result = [[0], [3, 4]]
    assert boolarr_to_sections(bools) == result

    bools = np.array([False, False, True, True, False, True])
    result = [[2, 3], [5]]
    assert boolarr_to_sections(bools) == result

    bools = np.array([False, False, True, True, False])
    result = [[2, 3]]
    assert boolarr_to_sections(bools) == result


def test_despike():
    sh = np.ones(100)
    sh[50] = 100
    sh[[49, 51]] = 3
    sh[80] = 200
    sh[[79, 81]] = 5
    result = np.ones(100)
    sampling_freq = 1024.0
    spike_threshold = 8.0
    spikes = detect_shear_spikes(
        sh,
        sampling_freq,
        spike_threshold,
        spike_include_before=10,
        spike_include_after=10,
    )

    spike_sections = boolarr_to_sections(spikes)
    spike_markers = sections_to_marker(spike_sections, len(sh))
    replace_spikes(
        sh,
        spike_markers,
        spike_replace_before=10,
        spike_replace_after=10,
    )

    assert np.all(result == sh)
