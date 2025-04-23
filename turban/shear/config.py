from netCDF4 import Dataset
from turban.config import SegmentConfig


class ShearConfig(SegmentConfig):
    freq_cutoff_antialias: float = 999.0
    freq_cutoff_corrupt: float = 999.0
    freq_highpass: float = 0.15  # [Hz]
    spatial_response_wavenum: float = 50.0  # [1/m]
    waveno_cutoff_spatial_corr: float = 999.0  # [1/m]
    spike_threshold: float = 8.0  # despiking in level 2
    max_tries: int = 10  # despiking in level 2
    spike_replace_before: int = 512  # despiking in level 2
    spike_replace_after: int = 512  # despiking in level 2
    spike_include_before: int = 10  # despiking in level 2
    spike_include_after: int = 20  # despiking in level 2
    cutoff_freq_lp: float = 0.5  # despiking in level 2

    @staticmethod
    def _attrs_from_atomix_netcdf(fname):
        attrs = super(ShearConfig, ShearConfig)._attrs_from_atomix_netcdf(fname)
        with Dataset(fname) as f:
            attrs.update(
                dict(
                    freq_highpass=f.HP_cut,
                )
            )
        return attrs
