from typing import ClassVar, Literal, cast
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
    Level,
    Processing,
    TimeseriesLevel,
)
from turban.utils.logging import get_logger
from turban.utils.seawater import diffusivity_temp, viscosity_kinematic

logger = get_logger(__name__)


@dataclass(kw_only=True)
class UTempLevel1(Level1):
    """Microtemperature Level 1: time derivative of temperature in physical units.

    ``spectral_correction`` is an optional multiplicative correction of the power
    spectra (e.g. for instrument electronics) on the frequency grid
    ``np.fft.rfftfreq(cfg.segment_length, 1 / cfg.sampfreq)``.
    """

    dtempdt: Float[ndarray, "ntemp time"]
    section_number: Int[ndarray, "time"]
    # this assumes that spectral_correction was evaluated on the frequency grid defined
    # by cfg.sampfrq and cfg.segment_length
    spectral_correction: Float[ndarray, "freq"] | None = None
    cfg: UTempConfig


@dataclass(kw_only=True)
class UTempLevel2(Level2[UTempLevel1]):
    """Microtemperature Level 2: cleaned temperature gradient

    TODO: implement a cleaning algorithm
    """

    dtempdt: Float[ndarray, "ntemp time"]
    cfg: UTempConfig

    level_below_cls: ClassVar[type[TimeseriesLevel]] = UTempLevel1

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
        logger.warning("No despiking implemented yet")
        kwarg.update(
            dict(
                time=level1.time,
                dtempdt=level1.dtempdt,
                senspeed=level1.senspeed,
                section_number=level1.section_number,
            )
        )
        return kwarg


@dataclass(kw_only=True)
class UTempLevel3(Level3[UTempLevel2]):
    """Microtemperature Level 3: chunk-wise temperature gradient spectra, derived
    from ``UTempLevel2``.
    """

    psi_k_gradt: Float[ndarray, "ntemp time waveno"]
    psi_noise: Float[ndarray, "ntemp waveno"]
    psi_f_gradt: Float[ndarray, "ntemp time waveno"]
    cfg: UTempConfig

    level_below_cls: ClassVar[type[TimeseriesLevel]] = UTempLevel2

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

        fp07 = cfg.fp07_response
        (
            waveno,
            psi_k,
            psi_f,
            freq,
            senspeed_avg,
            section_number_slow,
            psi_noise,
            _,
        ) = temperature_gradient_spectra(
            dtempdt=level2.dtempdt,
            senspeed=level2.senspeed,
            segment_length=cfg.segment_length,
            segment_overlap=cfg.segment_overlap,
            chunk_length=cfg.chunk_length,
            chunk_overlap=cfg.chunk_overlap,
            sampfreq=cfg.sampfreq,
            section_number=level2.section_number,
            spectral_correction=level2.level_below.spectral_correction,
            tau0=None if fp07 is None else fp07.tau0,
            gamma=None if fp07 is None else fp07.gamma,
            npoles=None if fp07 is None else fp07.npoles,
            response_model=None if fp07 is None else fp07.name,
        )

        kwarg.update(
            dict(
                time=agg_fast_to_slow(
                    level2.time,
                    section_number_or_data_len=level2.section_number,
                    chunk_length=cfg.chunk_length,
                    chunk_overlap=cfg.chunk_overlap,
                    agg_method="take_mid",
                ),
                psi_k_gradt=psi_k,
                psi_noise=psi_noise,
                waveno=waveno,
                psi_f_gradt=psi_f,
                freq=freq,
                senspeed=senspeed_avg,
                section_number=section_number_slow,
            )
        )
        return kwarg


@dataclass(kw_only=True)
class UTempLevel4(Level4):
    chi: Float[ndarray, "ntemp time"]
    eps: Float[ndarray, "ntemp time"]

    level_below_cls: ClassVar[type[TimeseriesLevel]] = UTempLevel3

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
                cfg=cfg,
            )
        )
        return kwarg


class UTempProcessing(Processing):

    _level_mapping: dict[Level, type[TimeseriesLevel]] = {
        Level.level1: UTempLevel1,
        Level.level2: UTempLevel2,
        Level.level3: UTempLevel3,
        Level.level4: UTempLevel4,
    }
