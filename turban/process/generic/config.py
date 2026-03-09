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

        Parameters
        ----------
        ds : xr.Dataset
            Dataset whose ``attrs`` will be updated in-place.
        """
        ds.attrs.update(self.model_dump())

    @classmethod
    def from_xarray(cls, ds: xr.Dataset):
        """Create from xarray Dataset global attributes.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset whose ``attrs`` contain the configuration fields.

        Returns
        -------
        SegmentConfig
            Validated instance constructed from the dataset attributes.
        """
        return cls.model_validate(ds.attrs)
