from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from turban.process.generic.api import (
    Level1,
    Level2,
    Level3,
    Level4,
    Processing,
)
from turban.process.generic.config import SegmentConfig
from turban.utils.util import agg_fast_to_slow

SAMPFREQ = 100.0
SEGMENT_LENGTH = 50
CHUNK_LENGTH = 100
CHUNK_OVERLAP = 50
N = 1000
SENSPEED = 0.5


def make_cfg():
    return SegmentConfig(
        sampfreq=SAMPFREQ,
        segment_length=SEGMENT_LENGTH,
        segment_overlap=25,
        chunk_length=CHUNK_LENGTH,
        chunk_overlap=CHUNK_OVERLAP,
    )


def make_level1():
    return Level1(
        time=np.arange(N, dtype=float),
        senspeed=SENSPEED * np.ones(N),
        section_number=np.ones(N, dtype=int),
        cfg=make_cfg(),
    )


def make_level2():
    return Level2(
        time=np.arange(N, dtype=float),
        senspeed=SENSPEED * np.ones(N),
        section_number=np.ones(N, dtype=int),
        cfg=make_cfg(),
    )


def expected_agg(arr, agg_method):
    return agg_fast_to_slow(
        arr,
        section_number_or_data_len=np.ones(N, dtype=int),
        chunk_length=CHUNK_LENGTH,
        chunk_overlap=CHUNK_OVERLAP,
        agg_method=agg_method,
    )


@pytest.mark.parametrize(
    "agg_method, name_out, expected_name",
    [
        ("mean", "foo", "foo"),
        ("median", "foo", "foo"),
        ("mean", None, "pres_mean"),
        ("median", None, "pres_median"),
    ],
)
def test_add_aux_data_name_out_level2_to_level3(agg_method, name_out, expected_name):
    pres = np.linspace(0.0, 100.0, N)
    level2 = make_level2()
    level2.add_aux_data(pres, name="pres", agg_method=agg_method, name_out=name_out)

    aux3 = Level3.from_level_below(level2)._aux_data

    assert list(aux3) == [expected_name]
    dims, arr = aux3[expected_name]
    assert dims == ["time"]
    np.testing.assert_allclose(arr, expected_agg(pres, agg_method))


def test_level2_from_level1():
    level1 = make_level1()
    level2 = Level2.from_level_below(level1)

    assert level2.level_below is level1
    assert level2.cfg is level1.cfg
    np.testing.assert_array_equal(level2.time, level1.time)
    np.testing.assert_array_equal(level2.senspeed, level1.senspeed)
    np.testing.assert_array_equal(level2.section_number, level1.section_number)


def test_level3_from_level2():
    level2 = make_level2()
    level3 = Level3.from_level_below(level2)

    assert level3.level_below is level2
    assert level3.cfg is level2.cfg

    np.testing.assert_array_equal(level3.time, expected_agg(level2.time, "take_mid"))
    np.testing.assert_allclose(level3.senspeed, expected_agg(level2.senspeed, "mean"))
    np.testing.assert_array_equal(
        level3.section_number, expected_agg(level2.section_number, "max")
    )

    freq = np.fft.rfftfreq(SEGMENT_LENGTH, d=1 / SAMPFREQ)
    np.testing.assert_allclose(level3.freq, freq)
    assert level3.waveno.shape == (len(level3.time), len(freq))
    np.testing.assert_allclose(
        level3.waveno,
        np.broadcast_to(freq / SENSPEED, level3.waveno.shape),
    )


def test_level4_from_level3():
    level3 = Level3.from_level_below(make_level2())
    level4 = Level4.from_level_below(level3)

    assert level4.level_below is level3
    assert level4.cfg is level3.cfg
    np.testing.assert_array_equal(level4.time, level3.time)
    np.testing.assert_array_equal(level4.section_number, level3.section_number)


def test_processing_chain_from_level1():
    level1 = make_level1()
    level1.add_aux_data(np.linspace(0.0, 100.0, N), name="pres")
    p = Processing(level1)

    assert p.level1 is level1
    assert isinstance(p.level2, Level2)
    assert isinstance(p.level3, Level3)
    assert isinstance(p.level4, Level4)
    assert p.level4.level_below is p.level3
    assert p.level3.level_below is p.level2
    assert p.level2.level_below is p.level1
    # all levels share the config of level 1
    for level in (p.level2, p.level3, p.level4):
        assert level.cfg is level1.cfg
    # aux data are aggregated at level 3 and read through by level 4
    assert "pres_mean" in p.level4.aux_data

    data_tree = p.to_xarray()
    outdir = Path("out/tests/process/generic")
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / "test_processing_chain_from_level1.nc"
    p.to_netcdf(str(fname))
    # check both the DataTree and the raw netCDF file (read without turban parsing)
    for tree in (data_tree, xr.load_datatree(fname)):
        assert set(tree.children) == {"level1", "level2", "level3", "level4"}
        # aux data are only stored at level 1 and 3
        assert "pres" in tree["level1"] and "pres" not in tree["level2"]
        assert "pres_mean" in tree["level3"] and "pres_mean" not in tree["level4"]

    # roundtrip parses the whole chain of levels
    p2 = Processing.from_xarray(data_tree)
    assert isinstance(p2.level1, Level1)
    assert p2.level4.aux_data["pres_mean"] is p2.level3.aux_data["pres_mean"]
    np.testing.assert_array_equal(p2.level4.time, p.level4.time)


def test_processing_subclass_requires_level_mapping():
    with pytest.raises(TypeError, match="_level_mapping"):

        class NoMapping(Processing):
            pass
