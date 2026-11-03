import pytest
import xarray as xr

from turban import ShearConfig, ShearLevel1, ShearProcessing
from turban.utils.filepaths import (
    atomix_benchmark_baltic_fpath,
    atomix_benchmark_faroe_fpath,
)

ATOMIX_GROUPS = ("L1_converted", "L2_cleaned", "L3_spectra", "L4_dissipation")


@pytest.mark.parametrize(
    "fpath, channels",
    [
        (atomix_benchmark_baltic_fpath, ("VIB1",)),
        (atomix_benchmark_faroe_fpath, ("ACC1", "ACC2", "ACC3")),
    ],
)
def test_vibration_channels_from_atomix_netcdf(fpath, channels):
    # channels are set together with their aux data, never by the config alone
    level1 = ShearLevel1.from_atomix_netcdf(fpath)
    assert level1.cfg.vibration_channels == channels
    assert set(channels) <= set(level1.aux_data)
    assert ShearConfig.from_atomix_netcdf(fpath).vibration_channels == ()


@pytest.mark.parametrize("group_without_vib", ["L1_converted", "L2_cleaned"])
@pytest.mark.parametrize("level", [1, 2])
def test_vibration_signals_in_one_group_only(tmp_path, group_without_vib, level):
    # copy of the Baltic benchmark file without VIB in one group
    fname = tmp_path / "baltic.nc"
    xr.Dataset(attrs=xr.open_dataset(atomix_benchmark_baltic_fpath).attrs).to_netcdf(
        fname
    )
    for group in ATOMIX_GROUPS:
        ds = xr.load_dataset(atomix_benchmark_baltic_fpath, group=group)
        if group == group_without_vib:
            ds = ds.drop_vars("VIB")
        ds.to_netcdf(fname, group=group, mode="a")

    p = ShearProcessing.from_atomix_netcdf(str(fname), level=level)
    assert p.cfg.vibration_channels == ("VIB1",)
    assert "VIB1" in p.level2.aux_data
