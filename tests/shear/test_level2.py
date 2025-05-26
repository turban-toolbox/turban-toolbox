import numpy as np
import matplotlib.pyplot as plt
from turban.process.shear.level2 import (
    boolarr_to_sections,
    sections_to_marker,
    detect_shear_spikes,
    replace_spike,
    replace_spikes,
    rollpad1,
    enlarge_bool,
    clean_shear,
    butterfilt,
)
from turban.process.shear.api import ShearLevel1, ShearLevel2
from tests.fixtures import atomix_nc_filename


def test_despike_benchmark(atomix_nc_filename):

    level1 = ShearLevel1.from_atomix_netcdf(atomix_nc_filename)
    ds1 = level1.to_xarray()

    # level2_bm = ShearLevel2.from_atomix_netcdf(atomix_nc_filename)
    # ds2_bm = level2_bm.to_xarray()

    # ds2 = ShearLevel2.from_level_below(level1).to_xarray()

    ti = slice(50_300, 50_800)
    # ti = slice(51_000, 51_500)
    # ti = slice(0, -1)
    # ti = slice(105_600, 106_000)
    # ti = slice(122_000, 125_000)
    # ti = slice(103_000, 107_000)

    isel = dict(n_shear=0, time=ti)
    # currently, no spike is detected in this segment!
    x = ds1.isel(**isel).shear.values
    assert np.any(detect_shear_spikes(x, 1024.0, 8.0, 512, 512, 0.5))


def test_despike_benchmark_plot(atomix_nc_filename):

    level1 = ShearLevel1.from_atomix_netcdf(atomix_nc_filename)
    cfg = level1.cfg
    ds1 = level1.to_xarray()
    level2_bm = ShearLevel2.from_atomix_netcdf(atomix_nc_filename)
    ds2_bm = level2_bm.to_xarray()

    i0 = 48_000  # start of segment
    ti = slice(i0, 55_800)  # for despiking
    tip0 = slice(50_300 - i0, 50_800 - i0)  # for plotting
    tip = slice(50_300, 50_800)  # for plotting

    sh = level1.shear[0, ti].copy()

    sampling_freq = cfg.sampling_freq

    segment_length = cfg.segment_length

    sh, ctr = clean_shear(
        sh,
        sampling_freq=sampling_freq,
        spike_threshold=8.0,
        max_tries=8,
        spike_replace_before=512,
        spike_replace_after=512,
        spike_include_before=10,
        spike_include_after=20,
        cutoff_freq_lp=0.5,
    )

    # after removal of spikes, can high-pass filter
    # Eq. 17
    sh_clean = butterfilt(
        signal=sh,
        cutoff_freq_Hz=0.5 / (segment_length / sampling_freq),
        sampling_freq=sampling_freq,
        btype="high",
    )

    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    plotarg = dict(marker=".", lw=0.1)
    isel = dict(n_shear=0, time=tip)

    ds1.isel(**isel).shear.plot(**plotarg, ax=ax)
    ds2_bm.isel(**isel).shear.plot(**plotarg, ax=ax)
    ax.plot(
        np.arange(len(sh[tip0])),
        sh[tip0],
        **plotarg,
    )
    ax.legend(["L1", "L2 benchmark", "L2 turban"])
    ax.grid()
    ax.set_title(f"Samples {tip.start}..{tip.stop}")
    fig.savefig("out/tests/level2-despike.png")


def test_replace_spike():
    sh = np.arange(10, dtype=float) / 100
    sh[4:6] = 1
    # replace data at indices [4, 5] with average over data
    # at indices [2, 3, 6, 7]
    r = (2 + 3 + 6 + 7) / 4.0 / 100
    replace_spike(sh, 4, 6, 2, 2)
    assert np.all(sh == np.array([0, 0.01, 0.02, 0.03, r, r, 0.06, 0.07, 0.08, 0.09]))


def test_replace_spikes():
    sh = np.arange(10, dtype=float) / 100
    sh[4:6] = 1
    spike_markers = np.zeros(10, dtype=int)
    spike_markers[4:6] = 1
    replace_spikes(sh, spike_markers, 2, 2)
    r = (2 + 3 + 6 + 7) / 4.0 / 100
    assert np.all(sh == np.array([0, 0.01, 0.02, 0.03, r, r, 0.06, 0.07, 0.08, 0.09]))


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
        cutoff_freq_lp=0.5,
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
