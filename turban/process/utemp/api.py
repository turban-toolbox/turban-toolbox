from typing import Literal, cast
from dataclasses import dataclass
from jaxtyping import Float, Int
from netCDF4 import Dataset
from numpy import newaxis, nan, ndarray
import numpy as np
import xarray as xr

from turban.utils.util import agg_fast_to_slow
from turban.process.utemp.config import UTempConfig

# from turban.process.utemp.level2 import process_level2
from turban.process.utemp.level3 import temperature_gradient_spectra
from turban.process.utemp.level4 import temperature_dissipation
from turban.process.generic.api import (
    AuxDataTypehintLevel12,
    AuxDataTypehintLevel34,
    Level1,
    Level2,
    Level3,
    Level4,
    Processing,
)

from turban.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(kw_only=True)
class UTempLevel1(Level1):
    dtempdt: Float[ndarray, "ntemp time"]
    cfg: UTempConfig


@dataclass(kw_only=True)
class UTempLevel2(Level2):
    senspeed: Float[ndarray, "time"]
    dtempdt: Float[ndarray, "ntemp time"]
    # num_despike_iter: Int[ndarray, "n_shear time"]

    @classmethod
    def _from_level_below_kwarg(
        cls,
        data: UTempLevel1,
    ):
        """Build constructor kwargs for UTempLevel2 from UTempLevel1 data.

        Parameters
        ----------
        data : UTempLevel1
            Level 1 microtemperature data.

        Returns
        -------
        dict
            Keyword arguments to pass to the UTempLevel2 constructor.
        """
        kwarg = super()._from_level_below_kwarg(data)
        level1 = data
        cfg = cast(UTempConfig, level1.cfg)  # just for type checkers to understand type
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
                dtempdt=utemp_cleaned,
                senspeed=level1.senspeed,
                # num_despike_iter=num_despike_iter,
                level_below=level1,
                cfg=cfg,
            )
        )
        return kwarg


@dataclass(kw_only=True)
class UTempLevel3(Level3):
    psi_k: Float[ndarray, "ntemp time freq"]
    psi_noise: Float[ndarray, "ntemp freq"]
    psi_f: Float[ndarray, "ntemp time freq"]

    @classmethod
    def _from_level_below_kwarg(
        cls,
        data: UTempLevel2,
    ) -> dict:
        """Build constructor kwargs for UTempLevel3 by computing temperature gradient spectra.

        Parameters
        ----------
        data : UTempLevel2
            Level 2 microtemperature data.

        Returns
        -------
        dict
            Keyword arguments to pass to the UTempLevel3 constructor.
        """
        level2 = data
        cfg = cast(UTempConfig, level2.cfg)
        kwarg = super()._from_level_below_kwarg(level2)

        (
            waveno,
            psi_k,
            psi_f,
            freq,
            senspeed_avg,
            section_number_slow,
            psi_noise,
            reshape_index,
        ) = temperature_gradient_spectra(
            dtempdt=level2.dtempdt,
            senspeed=level2.senspeed,
            segment_length=cfg.segment_length,
            segment_overlap=cfg.segment_overlap,
            chunk_length=cfg.chunk_length,
            chunk_overlap=cfg.chunk_overlap,
            sampfreq=cfg.sampfreq,
            waveno_limit_upper=cfg.waveno_limit_upper,
            diff_gain=cfg.diff_gain,
            section_number=level2.section_number,
        )

        kwarg.update(
            dict(
                time=agg_fast_to_slow(
                    level2.time,
                    section_number_or_data_len=level2.section_number,
                    chunk_length=level2.cfg.chunk_length,
                    chunk_overlap=level2.cfg.chunk_overlap,
                    agg_method="take_mid",
                ),
                psi_k=psi_k,
                psi_noise=psi_noise,
                waveno=waveno,
                psi_f=psi_f,
                freq=freq,
                senspeed=senspeed_avg,
                section_number=section_number_slow,
                level_below=level2,
            )
        )
        return kwarg


@dataclass(kw_only=True)
class UTempLevel4(Level4):
    chi: Float[ndarray, "ntemp time"]
    eps: Float[ndarray, "ntemp time"]

    @classmethod
    def _from_level_below_kwarg(
        cls,
        level3: UTempLevel3,
    ) -> dict:
        cfg = cast(UTempConfig, level3.cfg)
        kwarg = super()._from_level_below_kwarg(level3)

        chi, eps = temperature_dissipation(
            psi_k=level3.psi_k,
            waveno=level3.waveno,
            psi_noise=level3.psi_noise,
            waveno_limit_upper=cfg.waveno_limit_upper,
        )
        kwarg.update(
            dict(
                chi=chi,
                eps=eps,
                level_below=level3,
            )
        )
        return kwarg


class UTempProcessing(Processing):

    _level_mapping = {1: UTempLevel1, 2: UTempLevel2, 3: UTempLevel3, 4: UTempLevel4}
