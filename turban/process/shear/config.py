import numpy as np
from pydantic import field_validator
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

    @field_validator("vibration_channels", mode="before")
    @classmethod
    def _str_to_tuple(cls, v):
        # netCDF returns a single-element string attribute as a plain str,
        # so need to manually recreate a single-element tuple
        return (v,) if isinstance(v, str) else v

    @property
    def number_signals_vibration_removal(self) -> int:
        """N_V in the ATOMIX paper: number of vibration (and other) signals used to
        remove vibration-coherent noise from the shear spectra (Goodman et al. 2006).
        See Eqs. 16-17 in the ATOMIX paper.

        Equals the number of ``vibration_channels``, since level 3 cannot be
        built if any of them is missing from the aux data.
        """
        return len(self.vibration_channels)

    @property
    def log_psi_std(self) -> float:
        r"""sigma_{ln\Psi} in the ATOMIX paper (Eq. 17, Sec. 3.3.5)"""
        return np.sqrt(
            5
            / 4
            * (
                self.number_fft_windows_per_chunk
                - self.number_signals_vibration_removal
            )
            ** (-7 / 9)
        )

    @staticmethod
    def _attrs_from_atomix_netcdf(fname):
        # imported here since turban.process.shear.atomix imports ShearConfig
        from turban.process.shear.atomix import config_attrs

        return config_attrs(fname)
