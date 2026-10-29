import pytest
import xarray as xr

from turban.process.generic.config import SegmentConfig
from turban.process.shear.config import ShearConfig
from turban.utils.filepaths import (
    atomix_benchmark_baltic_fpath,
    atomix_benchmark_faroe_fpath,
)


def test_config():
    for path in [atomix_benchmark_baltic_fpath, atomix_benchmark_faroe_fpath]:
        for Config in [SegmentConfig, ShearConfig]:
            Config.from_atomix_netcdf(path)


@pytest.mark.parametrize("vibration_channels", [tuple(), ("ACC",), ("ACCx", "ACCy")])
def test_tuple_netcdf_roundtrip(tmp_path, vibration_channels):
    """Empty and single-element tuples must survive netCDF attribute storage."""
    cfg = ShearConfig(
        sampfreq=512.0,
        segment_length=1024,
        segment_overlap=512,
        chunk_length=4096,
        chunk_overlap=2048,
        vibration_channels=vibration_channels,
    )
    ds = xr.Dataset()
    cfg.add_to_xarray(ds)
    fpath = tmp_path / "config.nc"
    ds.to_netcdf(fpath)
    with xr.open_dataset(fpath) as ds_read:
        cfg_read = ShearConfig.from_xarray(ds_read)
    assert cfg_read.vibration_channels == vibration_channels
    assert cfg_read == cfg
