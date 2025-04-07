from logging import warnings
from typing import Literal
from dataclasses import dataclass
from jaxtyping import Float, Int
from .config import ShearConfig
from numpy import newaxis, nan, ndarray
import numpy as np
import xarray as xr

from turban.shear.level2 import process_level2
from turban.shear.level3 import process_level3
from turban.shear.level4 import process_level4


@dataclass
class ShearLevel1:
    pspd: Float[ndarray, "time"]
    shear: Float[ndarray, "n_shear time"]
    section_marker: Int[ndarray, "time"] | None
    cfg: ShearConfig

    @classmethod
    def from_atomix_netcdf(cls, fname: str):
        ds = xr.load_dataset(fname, group="L1_converted")
        ds2 = xr.load_dataset(fname, group="L2_cleaned")
        return cls(
            pspd=ds.PSPD_REL.values,
            shear=ds.SHEAR.values,
            section_marker=ds2["SECTION_NUMBER"].values.astype(int),
            cfg=ShearConfig.from_atomix_netcdf(fname),
        )


@dataclass
class ShearLevel2:
    shear: Float[ndarray, "n_shear time"]
    pspd: Float[ndarray, "time"]
    n_despiked: Int[ndarray, "n_shear time"] | None
    cfg: ShearConfig

    @classmethod
    def from_level1(
        cls,
        level1: ShearLevel1,
    ):
        sh_cleaned, n_despiked = process_level2(
            level1.shear,
            level1.section_marker,
            level1.cfg.sampling_freq,
            level1.cfg.fft_length,  # TODO: from own utility or user-supplied
        )

        return cls(
            shear=sh_cleaned,
            pspd=level1.pspd,
            n_despiked=n_despiked,
            cfg=level1.cfg,
        )

    @classmethod
    def from_atomix_netcdf(cls, fname: str):
        ds = xr.load_dataset(fname, group="L2_cleaned")
        return cls(
            shear=ds.SHEAR.values,
            pspd=ds.PSPD_REL.values,
            n_despiked=None,
            cfg=ShearConfig.from_atomix_netcdf(fname),
        )


@dataclass
class ShearLevel3:
    Pk: Float[ndarray, "nshear time wavenumber"]
    k: Float[ndarray, "time wavenumber"]
    Pf: Float[ndarray, "nshear time wavenumber"] | None
    freq: Float[ndarray, "wavenumber"] | None
    platform_speed: Float[ndarray, "time"]
    section_marker: Int[ndarray, "time"] | None
    cfg: ShearConfig

    @classmethod
    def from_level2(
        cls,
        level1: ShearLevel1,
        level2: ShearLevel2,
    ) -> "ShearLevel3":
        k, Pk, Pf, freq, platform_speed, ancillary = process_level3(
            shear=level2.shear,
            pspd=level2.pspd,
            section_marker=level1.section_marker,
            fft_length=level2.cfg.fft_length,
            sampling_freq=level2.cfg.sampling_freq,
            spatial_response_wavenum=level2.cfg.spatial_response_wavenum,
            freq_highpass=level2.cfg.freq_highpass,
            fft_overlap=level2.cfg.fft_overlap,
            diss_length=level2.cfg.diss_length,
            diss_overlap=level2.cfg.diss_overlap,
        )

        return cls(
            Pk=Pk,
            k=k,
            Pf=Pf,
            freq=freq,
            platform_speed=platform_speed,
            section_marker=None,
            cfg=level2.cfg,
        )

    @classmethod
    def from_atomix_netcdf(cls, fname: str):
        ds = xr.load_dataset(fname, group="L3_spectra").transpose(
            ..., "N_SHEAR_SENSORS", "TIME_SPECTRA", "WAVENUMBER"
        )

        return cls(
            Pk=ds["SH_SPEC"].values,
            k=ds["KCYC"].values,
            Pf=None,
            freq=None,
            platform_speed=ds["PSPD_REL"].values,
            section_marker=None,
            cfg=ShearConfig.from_atomix_netcdf(fname),
        )

    def to_xarray(self):
        return xr.Dataset(
            {
                "k": (["time_slow", "wavenumber"], self.k),
                "Pk": (["nshear", "time_slow", "wavenumber"], self.Pk),
                "Pf": (
                    (["nshear", "time_slow", "wavenumber"], self.Pf)
                    if self.Pf is not None
                    else None
                ),
                "freq": (["wavenumber"], self.freq) if self.freq is not None else None,
                "platform_speed": (["time_slow"], self.platform_speed),
            }
        )

    @property
    def number_signals_vibration_removal(self):
        """N_V in the ATOMIX paper"""
        warnings.warn("Not implemented")
        return 0

    @property
    def log_psi_var(self):
        """sigma^2_{ln\Psi} in the ATOMIX paper"""
        return (
            5
            / 4
            * (
                self.cfg.number_fft_windows_per_spectrum
                - self.number_signals_vibration_removal
            )
            ** (-7 / 9)
        )

    @property
    def Pk_confidence_interval(self) -> Float[ndarray, "2 time wavenumber"]:
        """95% confidence interval of power spectrum.
        Eq. 23 in the ATOMIX paper"""
        return np.concatenate(
            (
                self.Pk * np.exp(1.96 * self.log_psi_var)[newaxis, ...],
                self.Pk * np.exp(-1.96 * self.log_psi_var)[newaxis, ...],
            ),
            axis=0,
        )

    @property
    def data_length(self) -> Float[ndarray, "time"]:
        """l_\epsilon in ATOMIX paper"""
        tau_eps = self.cfg.diss_length / self.cfg.sampling_freq
        return tau_eps * self.platform_speed


@dataclass
class ShearLevel4:
    eps: Float[ndarray, "nshear time"]
    eps_source_flag: Int[ndarray, "nshear time"]
    log_diss_var: Float[ndarray, "nshear time"]
    kolm_length: Float[ndarray, "nshear time"]
    resolved_var_frac: Float[ndarray, "nshear time"]  # V_fin ATOMIX paper
    num_spec_points: Int[ndarray, "nshear time"]
    cfg: ShearConfig

    @classmethod
    def from_level3(
        cls,
        level3: ShearLevel3,
    ) -> "ShearLevel4":
        (
            eps,
            eps_source_flag,
            log_diss_var,
            kolm_length,
            resolved_var_frac,
            num_spec_points,
        ) = process_level4(
            psi=level3.Pk,
            wavenumber=level3.k,
            platform_speed=level3.platform_speed,
            waveno_cutoff_spatial_corr=level3.cfg.waveno_cutoff_spatial_corr,
            freq_cutoff_antialias=level3.cfg.freq_cutoff_antialias,
            freq_cutoff_corrupt=level3.cfg.freq_cutoff_corrupt,
            data_length=level3.data_length,
            log_psi_var=level3.log_psi_var,
        )
        return cls(
            eps=eps,
            eps_source_flag=eps_source_flag,
            log_diss_var=log_diss_var,
            kolm_length=kolm_length,
            resolved_var_frac=resolved_var_frac,
            num_spec_points=num_spec_points,
            cfg=level3.cfg,
        )

    @classmethod
    def from_atomix_netcdf(cls, fname: str) -> "ShearLevel4":
        with xr.open_dataset(fname, group="L4_dissipation") as ds:
            return cls(
                eps=ds["EPSI"].values,
                cfg=ShearConfig.from_atomix_netcdf(fname),
            )

    def to_xarray(self):
        return xr.Dataset(
            data_vars={
                "eps": (["nshear", "time_slow"], self.eps),
                # "eps_specint": (["nshear", "time_slow"], eps),
                # "eps_isrfit": (["nshear", "time_slow"], eps),
            }
        )


class ShearProcessing:

    def __init__(
        self,
        level1: ShearLevel1 | None,
        level2: ShearLevel2 | None,
        level3: ShearLevel3 | None,
        level4: ShearLevel4 | None,
    ):
        self._level1 = level1
        self._level2 = level2
        self._level3 = level3
        self._level4 = level4

    @classmethod
    def from_atomix_netcdf(
        cls,
        fname: str,
        load_levels: tuple[Literal[1, 2, 3, 4], ...] = (1, 2, 3, 4),
    ):
        # TODO figure out what to do with segment_marker
        _level1 = ShearLevel1.from_atomix_netcdf(fname) if 1 in load_levels else None
        _level2 = ShearLevel2.from_atomix_netcdf(fname) if 2 in load_levels else None
        _level3 = ShearLevel3.from_atomix_netcdf(fname) if 3 in load_levels else None
        _level4 = ShearLevel4.from_atomix_netcdf(fname) if 4 in load_levels else None
        return cls(_level1, _level2, _level3, _level4)

    @property
    def level1(self):
        if self._level1 is None:
            raise ValueError("Level 1 data not loaded")
        return self._level1

    @property
    def level2(self):
        if self._level2 is None:
            self._level2 = ShearLevel2.from_level1(self.level1)
        return self._level2

    @property
    def level3(self):
        if self._level3 is None:
            self._level3 = ShearLevel3.from_level2(self.level1, self.level2)
        return self._level3

    @property
    def level4(self):
        if self._level4 is None:
            self._level4 = ShearLevel4.from_level3(self.level3)
        return self._level4

    @property
    def cfg(self):
        configs = [
            l.cfg
            for l in [self._level1, self._level2, self._level3, self._level4]
            if l is not None
        ]
        assert all(cfg == configs[0] for cfg in configs), "Inconsistent configurations"
        return configs[0]
