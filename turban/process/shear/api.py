import warnings
from typing import Literal, cast
from dataclasses import dataclass
from jaxtyping import Float, Int
from netCDF4 import Dataset
from .config import ShearConfig
from numpy import newaxis, nan, ndarray
import numpy as np
import xarray as xr

from turban.utils.util import agg_fast_to_slow
from turban.process.shear.level2 import process_level2
from turban.process.shear.level3 import process_level3
from turban.process.shear.level4 import process_level4, get_quality_metric
from turban.process.generic.api import (
    AuxDataTypehintLevel12,
    AuxDataTypehintLevel34,
    Level1,
    Level2,
    Level3,
    Level4,
    Processing,
)


@dataclass(kw_only=True)
class ShearLevel1(Level1):
    shear: Float[ndarray, "nshear time"]
    section_number: Int[ndarray, "time"]
    cfg: ShearConfig

    @classmethod
    def from_atomix_netcdf(cls, fname: str):
        ds = xr.load_dataset(fname, group="L1_converted")
        # TODO: handle section_number through level 2
        ds2 = xr.load_dataset(fname, group="L2_cleaned")
        return cls(
            time=ds.TIME.values,
            senspeed=ds.PSPD_REL.values if "PSPD_REL" in ds else ds2.PSPD_REL.values,
            shear=ds.SHEAR.values,
            section_number=ds2["SECTION_NUMBER"].values.astype(int),
            cfg=cast(ShearConfig, ShearConfig.from_atomix_netcdf(fname)),
        )


@dataclass(kw_only=True)
class ShearLevel2(Level2):
    shear: Float[ndarray, "nshear time"]
    section_number: Int[ndarray, "time"]
    num_despike_iter: Int[ndarray, "nshear time"]
    cfg: ShearConfig

    @classmethod
    def _from_level_below_kwarg(
        cls,
        data: ShearLevel1,
    ):
        kwarg = super()._from_level_below_kwarg(data)
        level1 = data
        cfg = cast(ShearConfig, level1.cfg)  # just for type checkers to understand type
        sh_cleaned, num_despike_iter = process_level2(
            level1.shear,
            level1.section_number,
            cfg.sampfreq,
            cfg.segment_length,
            cfg.cutoff_freq_lp,
            cfg.spike_threshold,
            cfg.max_tries,
            cfg.spike_replace_before,
            cfg.spike_replace_after,
            cfg.spike_include_before,
            cfg.spike_include_after,
        )

        kwarg.update(
            dict(
                time=level1.time,
                shear=sh_cleaned,
                senspeed=level1.senspeed,
                section_number=level1.section_number,
                num_despike_iter=num_despike_iter,
                level_below=level1,
            )
        )
        return kwarg

    @classmethod
    def from_atomix_netcdf(cls, fname: str):
        ds = xr.load_dataset(fname, group="L2_cleaned")
        return cls(
            time=ds.TIME.values,
            shear=ds.SHEAR.values,
            senspeed=ds.PSPD_REL.values,
            # TODO: apparently not exported in benchmark files...?
            section_number=ds["SECTION_NUMBER"].values.astype(int),
            num_despike_iter=9999 * np.zeros_like(ds.SHEAR.values, dtype=int),
            level_below=ShearLevel1.from_atomix_netcdf(fname),
        )


@dataclass(kw_only=True)
class ShearLevel3(Level3):
    psi_k_sh: Float[ndarray, "nshear time waveno"]
    psi_f_sh: Float[ndarray, "nshear time waveno"]
    # TODO load from atomix netcdf
    spike_fraction: Float[ndarray, "nshear time"]
    max_despike_iter: Int[ndarray, "nshear time"]
    cfg: ShearConfig

    @classmethod
    def _from_level_below_kwarg(
        cls,
        data: ShearLevel2,
    ) -> dict:
        kwarg = super()._from_level_below_kwarg(data)
        k, psi_k_sh, psi_f_sh, freq, senspeed, section_number = process_level3(
            shear=data.shear,
            senspeed=data.senspeed,
            section_number=data.section_number,
            segment_length=data.cfg.segment_length,
            sampfreq=data.cfg.sampfreq,
            spatial_response_wavenum=data.cfg.spatial_response_wavenum,
            freq_highpass=data.cfg.freq_highpass,
            segment_overlap=data.cfg.segment_overlap,
            chunk_length=data.cfg.chunk_length,
            chunk_overlap=data.cfg.chunk_overlap,
        )

        spikes_per_chunk = agg_fast_to_slow(
            data.num_despike_iter > 0,  # has been despiked
            chunk_length=data.cfg.chunk_length,
            chunk_overlap=data.cfg.chunk_overlap,
            section_number_or_data_len=data.section_number,
            agg_method="sum",  # count number of occurences
        )

        spike_fraction = spikes_per_chunk / data.cfg.chunk_length

        max_despike_iter = agg_fast_to_slow(
            data.num_despike_iter,
            chunk_length=data.cfg.chunk_length,
            chunk_overlap=data.cfg.chunk_overlap,
            section_number_or_data_len=data.section_number,
            agg_method="max",
        )

        time_slow = agg_fast_to_slow(
            data.time,
            chunk_length=data.cfg.chunk_length,
            chunk_overlap=data.cfg.chunk_overlap,
            section_number_or_data_len=data.section_number,
            agg_method="take_mid",
        )
        kwarg.update(
            dict(
                time=time_slow,
                psi_k_sh=psi_k_sh,
                waveno=k,
                psi_f_sh=psi_f_sh,
                freq=freq,
                senspeed=senspeed,
                section_number=section_number,
                spike_fraction=spike_fraction,
                max_despike_iter=max_despike_iter,
                level_below=data,
            )
        )
        return kwarg

    @classmethod
    def from_atomix_netcdf(cls, fname: str):
        ds = xr.load_dataset(fname, group="L3_spectra").transpose(
            ..., "N_SHEAR_SENSORS", "TIME_SPECTRA", "WAVENUMBER"
        )

        return cls(
            time=ds["TIME"].values,
            psi_k_sh=ds["SH_SPEC"].values,
            waveno=ds["KCYC"].values,
            psi_f_sh=ds["SH_SPEC"].values * np.nan,
            freq=np.nan * np.ones(ds["KCYC"].values.shape[-1]),
            senspeed=ds["PSPD_REL"].values,
            section_number=ds["SECTION_NUMBER"].values.astype(int),
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
                self.cfg.number_fft_windows_per_chunk
                - self.number_signals_vibration_removal
            )
            ** (-7 / 9)
        )

    @property
    def psi_k_sh_confidence_interval(self) -> Float[ndarray, "2 time waveno"]:
        """95% confidence interval of power spectrum.
        Eq. 23 in the ATOMIX paper"""
        return np.concatenate(
            (
                self.psi_k_sh * np.exp(1.96 * self.log_psi_var)[newaxis, ...],
                self.psi_k_sh * np.exp(-1.96 * self.log_psi_var)[newaxis, ...],
            ),
            axis=0,
        )

    @property
    def data_length(self) -> Float[ndarray, "time"]:
        """l_\epsilon in ATOMIX paper"""
        tau_eps = self.cfg.chunk_length / self.cfg.sampfreq
        return tau_eps * self.senspeed


@dataclass(kw_only=True)
class ShearLevel4(Level4):
    eps: Float[ndarray, "nshear time"]
    eps_source_flag: Int[ndarray, "nshear time"]
    log_diss_var: Float[ndarray, "nshear time"]
    log_diss_mad: Float[ndarray, "nshear time"]
    kolmlen: Float[ndarray, "nshear time"]
    resolved_var_frac: Float[ndarray, "nshear time"]  # V_f in ATOMIX paper
    num_spec_points: Int[ndarray, "nshear time"]
    quality_metric: Int[ndarray, "nshear time"]
    cfg: ShearConfig

    @classmethod
    def _from_level_below_kwarg(
        cls,
        data: ShearLevel3,
    ) -> dict:
        kwarg = super()._from_level_below_kwarg(data)
        level3 = data
        (
            eps,
            eps_source_flag,
            log_diss_var,
            kolmlen,
            resolved_var_frac,
            fom,
            log_diss_mad,
            num_spec_points,
        ) = process_level4(
            psi=level3.psi_k_sh,
            waveno=level3.waveno,
            senspeed=level3.senspeed,
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

        kwarg.update(
            dict(
                time=level3.time,
                eps=eps,
                eps_source_flag=eps_source_flag,
                log_diss_var=log_diss_var,
                log_diss_mad=log_diss_mad,
                kolmlen=kolmlen,
                resolved_var_frac=resolved_var_frac,
                num_spec_points=num_spec_points,
                quality_metric=quality_metric,
                level_below=level3,
            )
        )
        return kwarg

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
        data_aux: AuxDataTypehintLevel12 | AuxDataTypehintLevel34 | None = None,
    ):
        """
        Create shear processing pipeline from ATOMIX netcdf file.

        Supplying data_aux here triggers the full API (dictionary with aggregation
        instructions if level<=2). If one wishes to use the simplified data aggregation
        API, one should first create a ShearLevelN object, then use .add_aux_data().
        """
        data = cls._level_mapping[level].from_atomix_netcdf(fname)
        data.add_aux_data(data_aux)
        return cls(data)


class NetcdfReader:
    """Load any netcdf with variable mapping from turban standard name (see variables.py)
    to name in the netcdf.
    NB This class is still under construction.

    TODO Load the variable mapping directly from variables.py."""

    def __init__(self, varmap: dict[str, str] | Literal["atomix"] | None = None):
        if varmap is None:
            self._map = {}
        elif varmap == "atomix":
            self._map = {
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
        else:
            self._map = varmap

    def read(self, fname: str, vars: list[str]) -> list[ndarray]:
        with Dataset(fname) as ds:
            data = [ds[self._map[var]][:] for var in vars]
        return data
