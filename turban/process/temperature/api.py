import warnings
from typing import Literal, cast
from dataclasses import dataclass
from jaxtyping import Float, Int
from netCDF4 import Dataset
from numpy import newaxis, nan, ndarray
import numpy as np
import xarray as xr

from turban.utils.util import agg_fast_to_slow, get_cleaned_fraction
from turban.process.temperature.config import TempConfig
# from turban.process.temperature.level2 import process_level2
from turban.process.temperature.level3 import temperature_gradient_spectra
from turban.process.temperature.level4 import temperature_dissipation
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
class TempLevel1(Level1):
    dtempdt: Float[ndarray, "ntemp time"]
    cfg: TempConfig


@dataclass(kw_only=True)
class TempLevel2(Level2):
    senspeed: Float[ndarray, "time"]
    dtempdt: Float[ndarray, "ntemp time"]
    # num_despike_iter: Int[ndarray, "n_shear time"]

    @classmethod
    def _from_level_below_kwarg(
        cls,
        data: TempLevel1,
    ):
        kwarg = super()._from_level_below_kwarg(data)
        level1 = data
        cfg = cast(TempConfig, level1.cfg)  # just for type checkers to understand type
        utemp_cleaned = level1.dtempdt
        # utemp_cleaned, num_despike_iter = process_level2(
        #     level1.utemp,
        #     level1.section_number,
        #     cfg.sampfreq,
        #     cfg.segment_length,
        # )

        kwarg.update(
            dict(
                time=level1.time,
                utemp=utemp_cleaned,
                senspeed=level1.senspeed,
                # num_despike_iter=num_despike_iter,
                level_below=level1,
            )
        )
        return kwarg


@dataclass(kw_only=True)
class TempLevel3(Level3):
    psi_k: Float[ndarray, "ntemp time freq"]
    psi_noise: Float[ndarray, "ntemp time freq"]
    psi_f: Float[ndarray, "ntemp time freq"]

    @classmethod
    def _from_level_below_kwarg(
        cls,
        data: TempLevel2,
    ) -> dict:
        level2 = data
        kwarg = super()._from_level_below_kwarg(level2)

        waveno, psi_k, psi_f, freq, senspeed_avg, section_number_slow, psi_noise = (
            temperature_gradient_spectra(
                dtempdt=level2.dtempdt,
                senspeed=level2.senspeed,
                segment_length=level2.cfg.segment_length,
                segment_overlap=level2.cfg.segment_overlap,
                chunk_length=level2.cfg.chunk_length,
                chunk_overlap=level2.cfg.chunk_overlap,
                sampfreq=level2.cfg.sampfreq,
                waveno_limit_upper=level2.cfg.waveno_limit_upper,
                section_number=level2.section_number,
            )
        )

        kwarg.update(
            dict(
                psi_k=psi_k,
                psi_noise=psi_noise,
                waveno=waveno,
                psi_f=psi_f,
                freq=freq,
                senspeed=senspeed_avg,
                section_number=section_number_slow,
                cfg=level2.cfg,
            )
        )
        return kwarg


@dataclass(kw_only=True)
class TempLevel4(Level4):
    chi: Float[ndarray, "ntemp time"]
    eps: Float[ndarray, "ntemp time"]

    @classmethod
    def _from_level_below_kwarg(
        cls,
        level3: TempLevel3,
    ) -> dict:
        cfg = cast(TempConfig, level3.cfg)
        kwarg = super()._from_level_below_kwarg(level3)

        chi, eps = temperature_dissipation(
            psi_k=level3.psi_k,
            waveno=level3.waveno,
            psi_noise=level3.psi_noise,
            waveno_limit_upper=cfg.waveno_limit_upper,
        )
        kwarg.update(dict(chi=chi, eps=eps, cfg=level3.cfg))
        return kwarg


class TempProcessing(Processing):

    _level_mapping = {1: TempLevel1, 2: TempLevel2, 3: TempLevel3, 4: TempLevel4}
