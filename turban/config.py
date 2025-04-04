from pydantic import BaseModel
from netCDF4 import Dataset

class SegmentConfig(BaseModel):
    """Configures segment-wise processing of timeseries."""
    sampling_freq: float # [Hz]
    fft_length: int
    fft_overlap: int
    diss_length: int
    diss_overlap: int

    @staticmethod
    def _attrs_from_atomix_netcdf(fname):
        with Dataset(fname) as f:
            return dict(
                sampling_freq=f.fs_fast,
                fft_length=int(f.fft_length),
                fft_overlap=int(f.fft_length / 2),
                diss_length=int(f.diss_length),
                diss_overlap=int(f.overlap),
            )

    @classmethod
    def from_atomix_netcdf(cls, fname):
        kwarg = cls._attrs_from_atomix_netcdf(fname)
        return cls(**kwarg)