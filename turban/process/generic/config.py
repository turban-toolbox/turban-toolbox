import json
import types
from typing import Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict
from netCDF4 import Dataset
import xarray as xr
import numpy as np

# netCDF attributes cannot hold None
NONE_FOR_NETCDF = "_NONE"


class SegmentConfig(BaseModel):
    """Configures segment-wise processing of timeseries."""

    # re-run validators (incl. model validators of subclasses) on attribute
    model_config = ConfigDict(validate_assignment=True)

    sampfreq: float  # [Hz]
    segment_length: int
    segment_overlap: int
    chunk_length: int
    chunk_overlap: int

    @staticmethod
    def _attrs_from_atomix_netcdf(fname):
        """Extract SegmentConfig attributes from an ATOMIX netCDF file.

        Parameters
        ----------
        fname : str
            Path to the ATOMIX netCDF file.

        Returns
        -------
        dict
            Configuration attributes with keys: ``sampfreq``, ``segment_length``,
            ``segment_overlap``, ``chunk_length``, ``chunk_overlap``.
        """
        with Dataset(fname) as f:
            return dict(
                sampfreq=f.fs_fast,
                segment_length=int(f.fft_length),
                segment_overlap=int(f.fft_length / 2),
                chunk_length=int(f.diss_length),
                chunk_overlap=int(f.overlap),
            )

    @classmethod
    def from_atomix_netcdf(cls, fname):
        """Create a SegmentConfig from an ATOMIX netCDF file.

        Parameters
        ----------
        fname : str
            Path to the ATOMIX netCDF file.

        Returns
        -------
        SegmentConfig
            New instance with attributes read from the file.
        """
        kwarg = cls._attrs_from_atomix_netcdf(fname)
        return cls(**kwarg)

    @property
    def number_fft_windows_per_chunk(self):
        """N_f in the ATOMIX paper"""
        fft_segment_start = np.arange(
            0,
            self.chunk_length - self.segment_length + 1,
            self.segment_length - self.segment_overlap,
        )  # start of each fft segment within a chunk
        return len(fft_segment_start)

    def add_to_xarray(self, ds: xr.Dataset):
        """Write configuration fields as global attributes of an xarray Dataset.

        Every field is written as one attribute. NetCDF attributes do not allow ``None``
        or nested models.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset whose ``attrs`` will be updated in-place.
        """
        for k in type(self).model_fields:
            v = getattr(self, k)
            if v is None:
                v = NONE_FOR_NETCDF
            elif isinstance(v, BaseModel):
                v = v.model_dump_json()
            ds.attrs[k] = v

    @classmethod
    def from_xarray(cls, ds: xr.Dataset):
        """Create from xarray Dataset global attributes.

        Inverse of ``add_to_xarray``.
        Nested models are parsed from JSON.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset whose ``attrs`` contain the configuration fields.

        Returns
        -------
        SegmentConfig
            Validated instance constructed from the dataset attributes.
        """
        attrs = {
            k: None if isinstance(v, str) and v == NONE_FOR_NETCDF else v
            for k, v in ds.attrs.items()
        }
        for k, field in cls.model_fields.items():
            if _nested_model(field.annotation) and isinstance(attrs.get(k), str):
                attrs[k] = json.loads(attrs[k])
        return cls.model_validate(attrs)


def _nested_model(annotation) -> type[BaseModel] | None:
    """Return the BaseModel class in ``annotation`` (e.g. ``Model | None``)."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) in (Union, types.UnionType):
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    return None
