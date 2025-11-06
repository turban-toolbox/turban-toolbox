from typing import Literal
from dataclasses import dataclass
from jaxtyping import Float, Int
from turban.process.temperature.config import TempConfig
from turban.utils.util import fft_grad
from numpy import nan, ndarray
import numpy as np
import xarray as xr
from turban.process.temperature.temperature import temperature_gradient_spectra


@dataclass(kw_only=True)
class TempLevel1:
    senspeed: Float[ndarray, "time"]
    temp: Float[ndarray, "ntemp time"]
    cfg: TempConfig


@dataclass(kw_only=True)
class TempLevel2:
    senspeed: Float[ndarray, "time"]
    dtemp_dt: Float[ndarray, "ntemp time"]
    # n_despiked: Int[ndarray, "ntemp time"] | None
    section_number: Int[ndarray, "time"] | None
    cfg: TempConfig

    @classmethod
    def from_level1(
        cls,
        level1: TempLevel1,
    ):
        dtemp_dt = fft_grad(level1.temp, 1 / level1.cfg.sampfreq)

        # cleaning routine for dtemp_dt

        return cls(
            dtemp_dt=dtemp_dt,
            senspeed=level1.senspeed,
            # n_despiked=n_despiked,
            cfg=level1.cfg,
        )


@dataclass(kw_only=True)
class TempLevel3:
    Pk: Float[ndarray, "ntemp time waveno"]
    k: Float[ndarray, "time waveno"]
    Pf: Float[ndarray, "ntemp time waveno"] | None
    freq: Float[ndarray, "waveno"] | None
    senspeed: Float[ndarray, "time"]
    section_number: Int[ndarray, "time"] | None
    cfg: TempConfig

    @classmethod
    def from_level2(
        cls,
        level1: TempLevel1,
        level2: TempLevel2,
    ) -> "TempLevel3":

        k, Pk, Pnoise = temperature_gradient_spectra(
            level2.dtemp_dt,
            level2.senspeed,
            level2.cfg.chunk_length,
            level2.cfg.chunk_overlap,
            level2.cfg.segment_length,
            level2.cfg.sampfreq,
        )

        k, Pk, Pf, freq, senspeed, ancillary = process_level3(
            shear=level2.shear,
            senspeed=level2.senspeed,
            section_number=level1.section_number,
            segment_length=level2.cfg.segment_length,
            sampfreq=level2.cfg.sampfreq,
            spatial_response_wavenum=level2.cfg.spatial_response_wavenum,
            freq_highpass=level2.cfg.freq_highpass,
            segment_overlap=level2.cfg.segment_overlap,
            chunk_length=level2.cfg.chunk_length,
            chunk_overlap=level2.cfg.chunk_overlap,
        )

        return cls(
            Pk=Pk,
            k=k,
            Pf=Pf,
            freq=freq,
            senspeed=senspeed,
            section_number=None,
            cfg=level2.cfg,
        )

    def to_xarray(self):
        return xr.Dataset(
            {
                "k": (["time_slow", "waveno"], self.k),
                "Pk": (["ntemp", "time_slow", "waveno"], self.Pk),
                "Pf": (
                    (["ntemp", "time_slow", "waveno"], self.Pf)
                    if self.Pf is not None
                    else None
                ),
                "freq": (["waveno"], self.freq) if self.freq is not None else None,
                "senspeed": (["time_slow"], self.senspeed),
            }
        )


@dataclass(kw_only=True)
class TempLevel4:
    eps: Float[ndarray, "ntemp time"]
    cfg: TempConfig

    @classmethod
    def from_level3(
        cls,
        level3: TempLevel3,
    ) -> "TempLevel4":
        eps, _, _ = process_level4(
            psi=level3.Pk,
            waveno=level3.k,
            senspeed=level3.senspeed,
            waveno_cutoff_spatial_corr=level3.cfg.waveno_cutoff_spatial_corr,
            freq_cutoff_antialias=level3.cfg.freq_cutoff_antialias,
            freq_cutoff_corrupt=level3.cfg.freq_cutoff_corrupt,
        )
        return cls(eps=eps, cfg=level3.cfg)

    @classmethod
    def from_atomix_netcdf(cls, fname: str) -> "TempLevel4":
        with xr.open_dataset(fname, group="L4_dissipation") as ds:
            return cls(
                eps=ds["EPSI"].values,
                cfg=TempConfig.from_atomix_netcdf(fname),
            )

    def to_xarray(self):
        return xr.Dataset(
            data_vars={
                "eps": (["ntemp", "time_slow"], self.eps),
                # "eps_specint": (["ntemp", "time_slow"], eps),
                # "eps_isrfit": (["ntemp", "time_slow"], eps),
            }
        )


class TempProcessing:

    def __init__(
        self,
        level1: TempLevel1 | None,
        level2: TempLevel2 | None,
        level3: TempLevel3 | None,
        level4: TempLevel4 | None,
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
        _level1 = TempLevel1.from_atomix_netcdf(fname) if 1 in load_levels else None
        _level2 = TempLevel2.from_atomix_netcdf(fname) if 2 in load_levels else None
        _level3 = TempLevel3.from_atomix_netcdf(fname) if 3 in load_levels else None
        _level4 = TempLevel4.from_atomix_netcdf(fname) if 4 in load_levels else None
        return cls(_level1, _level2, _level3, _level4)

    @property
    def level1(self):
        if self._level1 is None:
            raise ValueError("Level 1 data not loaded")
        return self._level1

    @property
    def level2(self):
        if self._level2 is None:
            self._level2 = TempLevel2.from_level1(self.level1)
        return self._level2

    @property
    def level3(self):
        if self._level3 is None:
            self._level3 = TempLevel3.from_level2(self.level1, self.level2)
        return self._level3

    @property
    def level4(self):
        if self._level4 is None:
            self._level4 = TempLevel4.from_level3(self.level3)
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
