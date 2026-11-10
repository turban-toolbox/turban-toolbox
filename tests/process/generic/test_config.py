import xarray as xr

from turban.process.generic.config import SegmentConfig


class MockConfig(SegmentConfig):
    """SegmentConfig with optional fields, to test handling of None attributes."""

    optional_arg: float | None = None
    optional_arg_set: float | None = None


def test_netcdf_roundtrip(tmp_path):
    """None cannot be stored as a netCDF attribute, so it must not be written, but be recovered upon loading."""
    cfg = MockConfig(
        sampfreq=512.0,
        segment_length=1024,
        segment_overlap=512,
        chunk_length=4096,
        chunk_overlap=2048,
        optional_arg=None,
        optional_arg_set=98.0,
    )
    ds = xr.Dataset()
    cfg.add_to_xarray(ds)
    assert "optional_arg" not in ds.attrs
    assert ds.attrs["optional_arg_set"] == 98.0
    assert all(v is not None for v in ds.attrs.values())
    fpath = tmp_path / "config.nc"
    ds.to_netcdf(fpath)
    with xr.open_dataset(fpath) as ds_read:
        cfg_read = MockConfig.from_xarray(ds_read)
    assert "optional_arg" not in ds_read.attrs
    assert cfg_read.optional_arg is None
    assert cfg_read == cfg
