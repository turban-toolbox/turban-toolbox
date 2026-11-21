import json

import pytest
from pydantic import ValidationError

from turban.process.generic.config import NONE_FOR_NETCDF

import xarray as xr

from turban.process.utemp.config import FP07_RESPONSE_MODELS, FP07Response, UTempConfig

SEGMENT_KWARGS = dict(
    sampfreq=1024.0,
    segment_length=512,
    segment_overlap=256,
    chunk_length=2048,
    chunk_overlap=1024,
)


@pytest.mark.parametrize("name", sorted(FP07_RESPONSE_MODELS))
def test_resolve_fp07_response_from_name(name):
    cfg = UTempConfig(**SEGMENT_KWARGS, fp07_response=name)
    assert cfg.fp07_response == FP07_RESPONSE_MODELS[name]


def test_resolve_fp07_response_unknown_name():
    with pytest.raises(ValidationError, match="Unknown FP07 response model"):
        UTempConfig(**SEGMENT_KWARGS, fp07_response="not_a_model")


@pytest.mark.parametrize(
    "fp07_response",
    [
        "nash99",
        FP07Response(name="custom", tau0=5e-3, gamma=-0.5, npoles=2),
        # relabelled: name is informative, values must win on read
        FP07Response(name="nash99", tau0=5e-3, gamma=-0.5, npoles=1),
        None,
    ],
)
def test_netcdf_roundtrip_fp07_response(tmp_path, fp07_response):
    cfg = UTempConfig(**SEGMENT_KWARGS, fp07_response=fp07_response)
    ds = xr.Dataset()
    cfg.add_to_xarray(ds)
    assert not any(isinstance(v, dict) for v in ds.attrs.values())
    if fp07_response is None:
        assert ds.attrs["fp07_response"] == NONE_FOR_NETCDF
    else:
        assert json.loads(ds.attrs["fp07_response"]) == cfg.fp07_response.model_dump()
    fpath = tmp_path / "config.nc"
    ds.to_netcdf(fpath)
    with xr.open_dataset(fpath) as ds_read:
        cfg_read = UTempConfig.from_xarray(ds_read)
    assert cfg_read == cfg

