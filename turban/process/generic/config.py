from pydantic import BaseModel
from netCDF4 import Dataset
import xarray as xr
import numpy as np


class SegmentConfig(BaseModel):
    """Configures segment-wise processing of timeseries."""

    sampfreq: float  # [Hz]
    segment_length: int
    segment_overlap: int
    chunk_length: int
    chunk_overlap: int

    @staticmethod
    def _attrs_from_atomix_netcdf(fname):
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
        ds.attrs.update(self.model_dump())
        # another option that would allow setting further attributes from variables.py:
        # (would require corresponding changes in from_xarray)
        # if hasattr(self, "cfg"):
        #     # set cfg as 0-dim data variables
        #     for key, val in self.cfg.model_dump().items():
        #         ds[key] = ([], val)

    @classmethod
    def from_xarray(cls, ds: xr.Dataset):
        return cls.model_validate(ds.attrs)
