from typing import ClassVar, cast
from dataclasses import dataclass
from jaxtyping import Float, Int
from numpy import ndarray
import numpy as np

from turban.utils.util import agg_fast_to_slow
from turban.process.utemp.config import UTempConfig
from turban.process.utemp.level3 import temperature_gradient_spectra
from turban.process.utemp.level4 import temperature_dissipation
from turban.process.generic.api import (
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
    section_number: Int[ndarray, "time"]
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
class UTempLevel4(Level4[UTempLevel3]):
    """Microtemperature Level 4: dissipation rates of temperature variance (chi) and
    TKE (eps) from a maximum likelihood spectral fit, derived from ``UTempLevel3``.

    Viscosity and diffusivity are computed from the auxiliary variable ``temp`` if
    present, otherwise ``cfg.molvisc_fallback`` and ``cfg.moldiff_fallback`` are
    used.
    """

    chi: Float[ndarray, "ntemp time"]
    eps: Float[ndarray, "ntemp time"]
    eps_lower: Float[ndarray, "ntemp time"]
    eps_upper: Float[ndarray, "ntemp time"]
    waveno_batchelor: Float[ndarray, "ntemp time"]
    waveno_batchelor_ci: Float[ndarray, "ntemp time"]
    likelihood_ratio: Float[ndarray, "ntemp time"]
    mad: Float[ndarray, "ntemp time"]
    mad_powerlaw: Float[ndarray, "ntemp time"]
    molvisc: Float[ndarray, "time"]
    moldiff: Float[ndarray, "time"]
    cfg: UTempConfig

    level_below_cls: ClassVar[type[TimeseriesLevel]] = UTempLevel3

    @classmethod
    def _from_level_below_kwarg(
        cls,
        data: UTempLevel3,
    ) -> dict:
        kwarg = super()._from_level_below_kwarg(data)
        level3 = data
        cfg = cast(UTempConfig, level3.cfg)

        # so linters understand we have a dict after __post_init__
        aux_data = cast(dict, level3._aux_data)
        ntime = len(level3.time)
        if "temp" in aux_data:
            # level 3 aux data entries are (dims, arr, attrs)
            temp = aux_data["temp"][1]
            molvisc = viscosity_kinematic(temp)
            moldiff = diffusivity_temp(temp)
            logger.debug("Calculated molecular viscosity and diffusivity from `temp`")
        else:
            molvisc = cfg.molvisc_fallback
            moldiff = cfg.moldiff_fallback
            logger.debug(
                f"Used fallback values {molvisc}, {moldiff} for molecular viscosity "
                "and diffusivity"
            )
        # always store as array so that it is exported to xarray
        molvisc = np.broadcast_to(molvisc, (ntime,)).astype(float)
        moldiff = np.broadcast_to(moldiff, (ntime,)).astype(float)

        (
            chi,
            eps,
            eps_lower,
            eps_upper,
            waveno_batchelor,
            waveno_batchelor_ci,
            likelihood_ratio,
            mad,
            mad_powerlaw,
        ) = temperature_dissipation(
            psi_k_gradt=level3.psi_k_gradt,
            waveno=level3.waveno,
            psi_noise=level3.psi_noise,
            waveno_limit_upper=(
                np.inf if cfg.waveno_limit_upper is None else cfg.waveno_limit_upper
            ),
            molvisc=molvisc,
            moldiff=moldiff,
            spectrum_model=cfg.spectrum_model,
            q=cfg.q,
            n_trial=100,
            dof=cfg.dof_spectrum,
        )
        kwarg.update(
            dict(
                time=level3.time,
                chi=chi,
                eps=eps,
                eps_lower=eps_lower,
                eps_upper=eps_upper,
                waveno_batchelor=waveno_batchelor,
                waveno_batchelor_ci=waveno_batchelor_ci,
                likelihood_ratio=likelihood_ratio,
                mad=mad,
                mad_powerlaw=mad_powerlaw,
                molvisc=molvisc,
                moldiff=moldiff,
            )
        )
        return kwarg


class UTempProcessing(Processing):
    """Microtemperature processing pipeline from any start level up to ``UTempLevel4``.
    """

    _level_mapping: dict[Level, type[TimeseriesLevel]] = {
        Level.level1: UTempLevel1,
        Level.level2: UTempLevel2,
        Level.level3: UTempLevel3,
        Level.level4: UTempLevel4,
    }
