from pydantic import BaseModel
from netCDF4 import Dataset
import numpy as np


class SegmentConfig(BaseModel):
    """Configures segment-wise processing of timeseries."""

    sampling_freq: float  # [Hz]
    segment_length: int
    segment_overlap: int
    diss_length: int
    diss_overlap: int

    @staticmethod
    def _attrs_from_atomix_netcdf(fname):
        with Dataset(fname) as f:
            return dict(
                sampling_freq=f.fs_fast,
                segment_length=int(f.fft_length),
                segment_overlap=int(f.fft_length / 2),
                diss_length=int(f.diss_length),
                diss_overlap=int(f.overlap),
            )

    @classmethod
    def from_atomix_netcdf(cls, fname):
        kwarg = cls._attrs_from_atomix_netcdf(fname)
        return cls(**kwarg)

    @property
    def number_fft_windows_per_spectrum(self):
        """N_f in the ATOMIX paper"""
        fft_segment_start = np.arange(
            0,
            self.diss_length - self.segment_length + 1,
            self.segment_length - self.segment_overlap,
        )  # start of each fft segment within a spectrum segment
        return len(fft_segment_start)
