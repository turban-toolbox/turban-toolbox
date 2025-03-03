# import pydantic # maybe later
from dataclasses import dataclass
from netCDF4 import Dataset


@dataclass
class ShearConfig:
    chunklen: int
    chunkoverlap: int
    sampling_freq: float
    fftlen: int = None
    freq_cutoff_antialias: float = 999.0
    freq_cutoff_corrupt: float = 999.0
    freq_highpass: float = 0.15
    sampling_freq: float = None
    spatial_response_wavenum: float = 50.0
    waveno_cutoff_spatial_corr: float = 999.0

    @classmethod
    def from_atomix_netcdf(cls, fname):
        with Dataset(fname) as f:
            diss_length = f.diss_length
            fftlen = int(f.fft_length)
            overlap = int(f.overlap)
            (n_fft_segments,) = f["L3_spectra/N_FFT_SEGMENTS"][:]

            chunklen = int(n_fft_segments)  # TODO: get chunklen from f
            chunkoverlap = (
                overlap / fftlen
            )  # TODO can this be non-integer? can turban accomodate this?

            return cls(
                sampling_freq=f.fs_fast,
                fftlen=fftlen,
                freq_highpass=f.HP_cut,
                chunklen=5,
                chunkoverlap=2,  # TODO
            )
