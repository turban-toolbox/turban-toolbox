from netCDF4 import Dataset
from turban.process.generic.config import SegmentConfig


class ShearConfig(SegmentConfig):
    freq_cutoff_antialias: float | None = None
    freq_cutoff_corrupt: float | None = None  #
    freq_highpass: float = 0.15  # [Hz]
    spatial_response_wavenum: float = 50.0  # [1/m]
    waveno_cutoff_spatial_corr: float | None = None  # [1/m] #
    waveno_spectral_min: float | None = None
    spike_threshold: float = 8.0  # despiking in level 2
    max_tries: int = 10  # despiking in level 2
    spike_replace_before: int = 512  # despiking in level 2
    spike_replace_after: int = 512  # despiking in level 2
    spike_include_before: int = 10  # despiking in level 2
    spike_include_after: int = 20  # despiking in level 2
    cutoff_freq_lp: float = 0.5  # despiking in level 2
    molvisc_fallback: float = 1.6e-6  # used for level4 when not available from level3
    # tuple of variable names in auxiliary level 2 data used for vibration removal
    vibration_channels: tuple[str, ...] = tuple()

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
