# import pydantic # maybe later
from dataclasses import dataclass
from netCDF4 import Dataset


@dataclass
class ShearConfig:
    sampling_freq: float # [Hz]
    fft_length: int
    fft_overlap: int
    diss_length: int
    diss_overlap: int
    freq_cutoff_antialias: float = 999.0
    freq_cutoff_corrupt: float = 999.0
    freq_highpass: float = 0.15 # [Hz]
    spatial_response_wavenum: float = 50.0 # [1/m]
    waveno_cutoff_spatial_corr: float = 999.0 # [1/m]

    @classmethod
    def from_atomix_netcdf(cls, fname):
        with Dataset(fname) as f:
            overlap = int(f.overlap)
            (n_fft_segments,) = f["L3_spectra/N_FFT_SEGMENTS"][:]

            return cls(
                sampling_freq=f.fs_fast,
                fft_length=int(f.fft_length),
                fft_overlap=int(f.fft_length / 2),
                diss_length=int(f.diss_length),
                diss_overlap=int(f.overlap),
                freq_highpass=f.HP_cut,
            )
