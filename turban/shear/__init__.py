from logging import warnings
from typing import Literal
from dataclasses import dataclass
from jaxtyping import Float, Int
from netCDF4 import Dataset
from .config import ShearConfig
from numpy import newaxis, nan, ndarray
import numpy as np
import xarray as xr

from turban.util import agg_fast_to_slow, get_cleaned_fraction
from turban.shear.level2 import process_level2
from turban.shear.level3 import process_level3
from turban.shear.level4 import process_level4, get_quality_metric
from turban.api import AggAux, Level1, Level2, Level3, Level4, Processing


@dataclass(kw_only=True)
class ShearLevel1(Level1):
    shear: Float[ndarray, "n_shear time"]
    section_marker: Int[ndarray, "time"]

    @classmethod
    def from_atomix_netcdf(cls, fname: str):
        ds = xr.load_dataset(fname, group="L1_converted")
        # TODO: handle section_marker through level 2
        ds2 = xr.load_dataset(fname, group="L2_cleaned")
        return cls(
            time=ds.TIME.values.astype(float),
            pspd=ds.PSPD_REL.values,
            shear=ds.SHEAR.values,
            section_marker=ds2["SECTION_NUMBER"].values.astype(int),
            cfg=ShearConfig.from_atomix_netcdf(fname),
        )


@dataclass(kw_only=True)
class ShearLevel2(Level2):
    shear: Float[ndarray, "n_shear time"]
    num_despike_iter: Int[ndarray, "n_shear time"]

    @classmethod
    def from_level_below(
        cls,
        data: ShearLevel1,
    ):
        level1 = data
        sh_cleaned, num_despike_iter = process_level2(
            level1.shear,
            level1.section_marker,
            level1.cfg.sampling_freq,
            level1.cfg.fft_length,  # TODO: from own utility or user-supplied
        )

        return cls(
            time=level1.time,
            shear=sh_cleaned,
            pspd=level1.pspd,
            num_despike_iter=num_despike_iter,
            level_below=level1,
        )

    @classmethod
    def from_atomix_netcdf(cls, fname: str):
        ds = xr.load_dataset(fname, group="L2_cleaned")
        return cls(
            time=ds.TIME.values.astype(float),
            shear=ds.SHEAR.values,
            pspd=ds.PSPD_REL.values,
            # TODO: apparently not exported in benchmark files...?
            num_despike_iter=9999 * np.zeros_like(ds.SHEAR.values, dtype=int),
            level_below=ShearLevel1.from_atomix_netcdf(fname),
        )


@dataclass(kw_only=True)
class ShearLevel3(Level3):
    Pk: Float[ndarray, "nshear time wavenumber"]
    Pf: Float[ndarray, "nshear time wavenumber"]
    # TODO load from atomix netcdf
    section_marker: Int[ndarray, "time"]
    spike_fraction: Float[ndarray, "nshear time"]
    max_despike_iter: Int[ndarray, "nshear time"]

    @classmethod
    def from_level_below(
        cls,
        data: ShearLevel2,
    ) -> "ShearLevel3":
        level2 = data
        level1 = data.level_below
        k, Pk, Pf, freq, platform_speed, section_marker = process_level3(
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

        spike_fraction = get_cleaned_fraction(
            x=level1.shear,
            x_clean=level2.shear,
            data_len=level1.shear.shape[-1],
            fft_length=level2.cfg.fft_length,
            fft_overlap=level2.cfg.fft_overlap,
            diss_length=level2.cfg.diss_length,
            diss_overlap=level2.cfg.diss_overlap,
            section_marker=level1.section_marker,
        )

        max_despike_iter = agg_fast_to_slow(
            level2.num_despike_iter,
            data_len=level2.num_despike_iter.shape[-1],
            fft_length=level2.cfg.fft_length,
            fft_overlap=level2.cfg.fft_overlap,
            diss_length=level2.cfg.diss_length,
            diss_overlap=level2.cfg.diss_overlap,
            section_marker=level1.section_marker,
            agg_method="max",
        )

        return cls(
            time=np.ones_like(platform_speed),  # TODO get from level 2
            Pk=Pk,
            waveno=k,
            Pf=Pf,
            freq=freq,
            platform_speed=platform_speed,
            section_marker=section_marker,
            spike_fraction=spike_fraction,
            max_despike_iter=max_despike_iter,
            level_below=data,
        )

    @classmethod
    def from_atomix_netcdf(cls, fname: str):
        ds = xr.load_dataset(fname, group="L3_spectra").transpose(
            ..., "N_SHEAR_SENSORS", "TIME_SPECTRA", "WAVENUMBER"
        )

        return cls(
            time=ds["TIME"].values.astype(float),
            Pk=ds["SH_SPEC"].values,
            waveno=ds["KCYC"].values,
            Pf=ds["SH_SPEC"].values * np.nan,
            freq=np.nan * np.ones(ds["KCYC"].values.shape[-1]),
            platform_speed=ds["PSPD_REL"].values,
            section_marker=ds["SECTION_NUMBER"].values.astype(int),
            spike_fraction=np.nan * np.ones_like(ds["SH_SPEC"].values[:, :, 0]),
            max_despike_iter=9999
            * np.ones_like(ds["SH_SPEC"].values[:, :, 0], dtype=int),
            level_below=ShearLevel2.from_atomix_netcdf(fname),
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


@dataclass(kw_only=True)
class ShearLevel4(Level4):
    eps: Float[ndarray, "nshear time"]
    eps_source_flag: Int[ndarray, "nshear time"]
    log_diss_var: Float[ndarray, "nshear time"]
    log_diss_mad: Float[ndarray, "nshear time"]
    kolm_length: Float[ndarray, "nshear time"]
    resolved_var_frac: Float[ndarray, "nshear time"]  # V_f in ATOMIX paper
    num_spec_points: Int[ndarray, "nshear time"]
    quality_metric: Int[ndarray, "nshear time"]

    @classmethod
    def from_level_below(
        cls,
        data: ShearLevel3,
    ) -> "ShearLevel4":
        level3 = data
        (
            eps,
            eps_source_flag,
            log_diss_var,
            kolm_length,
            resolved_var_frac,
            fom,
            log_diss_mad,
            num_spec_points,
        ) = process_level4(
            psi=level3.Pk,
            wavenumber=level3.waveno,
            platform_speed=level3.platform_speed,
            waveno_cutoff_spatial_corr=level3.cfg.waveno_cutoff_spatial_corr,
            freq_cutoff_antialias=level3.cfg.freq_cutoff_antialias,
            freq_cutoff_corrupt=level3.cfg.freq_cutoff_corrupt,
            data_length=level3.data_length,
            log_psi_var=level3.log_psi_var,
        )

        quality_metric = get_quality_metric(
            eps=eps,
            eps_source_flag=eps_source_flag,
            fom=fom,
            spike_fraction=level3.spike_fraction,
            log_diss_var=log_diss_var,
            num_spec_points=num_spec_points,
            num_despike_iter=level3.max_despike_iter,
            resolved_var_frac=resolved_var_frac,
        )

        return cls(
            time=np.ones_like(level3.platform_speed),  # TODO get from level 2
            eps=eps,
            eps_source_flag=eps_source_flag,
            log_diss_var=log_diss_var,
            log_diss_mad=log_diss_mad,
            kolm_length=kolm_length,
            resolved_var_frac=resolved_var_frac,
            num_spec_points=num_spec_points,
            quality_metric=quality_metric,
            level_below=level3,
        )

    @classmethod
    def from_atomix_netcdf(cls, fname: str) -> "ShearLevel4":
        # TODO: flag to switch off loading of levels below
        with xr.open_dataset(fname, group="L4_dissipation") as ds:
            return cls(
                # TODO fix arguments
                eps=ds["EPSI"].values,
                level_below=ShearLevel3.from_atomix_netcdf(fname),
            )


class ShearProcessing(Processing):

    _level_mapping = {1: ShearLevel1, 2: ShearLevel2, 3: ShearLevel3, 4: ShearLevel4}

    @classmethod
    def from_atomix_netcdf(
        cls,
        fname: str,
        level: Literal[1, 2, 3, 4],
    ):
        data = cls._level_mapping[level].from_atomix_netcdf(fname)

        aux_vars = ["time", "press", "temp", "cond"]
        arr = dict(zip(aux_vars, AtomixNetcdfLoader().load(fname, aux_vars)))
        data_aux = {
            "time": (
                ["time"],
                arr["time"],
                {"mean": "time_slow"},
            ),
            "press": (
                ["time"],
                arr["press"],
                {"mean": "press"},
            ),
            "temp": (
                ["time"],
                arr["temp"][0, :],
                {"mean": "temp"},
            ),
            "cond": (
                ["time"],
                arr["cond"],
                {"mean": "cond"},
            ),
        }
        coords_aux = ["time", "time_slow"]
        return cls(data, level, data_aux, coords_aux)


class AtomixNetcdfLoader:
    _map = {
        "time": "L1_converted/TIME",
        # 'L1_converted/SHEAR',
        # 'L1_converted/TIME_CTD',
        # 'L1_converted/PSPD_REL',
        "press": "L1_converted/PRES",
        # 'L1_converted/VIB',
        "temp": "L1_converted/TEMP",
        # 'L1_converted/TEMP_CTD',
        "cond": "L1_converted/CNDC",
    }

    def load(self, fname: str, vars: list[str]):
        with Dataset(fname) as ds:
            for var in vars:
                print(ds[self._map[var]][:])
            data = [ds[self._map[var]][:] for var in vars]
        return data
