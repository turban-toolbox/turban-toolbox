from netCDF4 import Dataset
from turban.config import SegmentConfig

class ShearConfig(SegmentConfig):
    freq_cutoff_antialias: float = 999.0
    freq_cutoff_corrupt: float = 999.0
    freq_highpass: float = 0.15 # [Hz]
    spatial_response_wavenum: float = 50.0 # [1/m]
    waveno_cutoff_spatial_corr: float = 999.0 # [1/m]

    @staticmethod
    def _attrs_from_atomix_netcdf(fname):
        attrs = super(ShearConfig, ShearConfig)._attrs_from_atomix_netcdf(fname)
        with Dataset(fname) as f:
            attrs.update(dict(
                freq_highpass=f.HP_cut,
            ))
        return attrs


