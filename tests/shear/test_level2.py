import numpy as np
import matplotlib.pyplot as plt
from turban.shear.level2 import (
    boolarr_to_sections,
    sections_to_marker,
    detect_shear_spikes,
    replace_spikes,
    rollpad1,
    enlarge_bool,
)
from turban.shear import ShearLevel1, ShearLevel2
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
    assert np.any(detect_shear_spikes(x, 1024.0, 8.0))


def test_despike_benchmark_plot(atomix_nc_filename):

    level1 = ShearLevel1.from_atomix_netcdf(atomix_nc_filename)
    ds1 = level1.to_xarray()

    level2_bm = ShearLevel2.from_atomix_netcdf(atomix_nc_filename)
    ds2_bm = level2_bm.to_xarray()

    ds2 = ShearLevel2.from_level_below(level1).to_xarray()

    ti = slice(50_300, 50_800)
    # ti = slice(51_000, 51_500)
    # ti = slice(0, -1)
    # ti = slice(105_600, 106_000)
    # ti = slice(122_000, 125_000)
    # ti = slice(103_000, 107_000)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    plotarg = dict(marker=".", lw=0.1, ax=ax)
    isel = dict(n_shear=0, time=ti)

    ds1.isel(**isel).shear.plot(**plotarg)
    ds2.isel(**isel).shear.plot(**plotarg)
    ds2_bm.isel(**isel).shear.plot(**plotarg)
    ax.legend(["L1", "L2 turban", "L2 benchmark"])
    ax.grid()
    ax.set_title(f"Samples {ti.start}..{ti.stop}")
    fig.savefig("out/tests/level2-despike.png")


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
