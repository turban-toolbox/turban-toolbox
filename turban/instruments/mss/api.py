import os
from pathlib import Path

import numpy as np
from numpy import newaxis
from jaxtyping import Num, Int, Float
from numpy import ndarray
import pandas as pd


from turban.utils.ctd import calc_ctd, fofonoff_filt
from turban.instruments.generic.config import InstrumentConfig
from turban.process.temperature.temperature import deconvolute_mss_ntchp
from turban.utils.util import channel_mapping

from turban.instruments.generic.api import Dropsonde
from turban.process.shear.api import ShearLevel1, ShearProcessing
from turban.process.shear.config import ShearConfig
from turban.utils.util import get_vsink, fft_grad
from turban.instruments.mss.mss_mrd import mrd


class MSS(Dropsonde):

    def __init__(self, cfg: InstrumentConfig | None = None):
        """Allow neglecting InstrumentConfig for now."""
        if cfg is not None:
            super().__init__(cfg)
        else:

            class MockCfg:
                sampfreq = 1024.0

            self.cfg = MockCfg()

    def read_mrd(self, fname: Path | str) -> None:
        self.mrd = mrd(str(fname))  # TODO make mrd uppercase (PEP8)

    def to_shear_level1(
        self, section_number: Int[ndarray, "time"], cfg: ShearConfig
    ) -> ShearLevel1:
        pressure_raw = self.mrd.level0["PRESS"]
        shear_raw = np.concatenate(
            [
                vals[newaxis]
                for chname, vals in self.mrd.level0.items()
                if chname.startswith("SHE")
            ],
            axis=0,
        )
        senspeed, pressure_lp = get_vsink(pressure_raw, self.cfg.sampfreq)
        shear_phys = mss_shear_physical(senspeed, shear_raw, self.cfg.sampfreq)
        return ShearLevel1(
            senspeed=senspeed,
            shear=shear_phys,
            section_number=section_number,
            cfg=cfg,
        )

    def to_shear_processing(
        self,
        section_number: Int[ndarray, "time"],
        cfg: ShearConfig,
    ) -> ShearProcessing:
        """Convert to shear processing pipeline."""
        return ShearProcessing(
            self.to_shear_level1(section_number, cfg),
            None,
            None,
            None,
        )


def level1(
    raw: Int[ndarray, "time 16"],
    probeconf_fname: str,
    lon: float,
    lat: float,
    sampfreq: float = 1024.0,
) -> dict[str, Num[ndarray, "time"]]:
    """
    Convert raw MSS data to physical units
    """
    temp_fast_channel, temp_emph_channel = channel_mapping(
        probeconf_fname, "TEMP_FAST", "TEMP_EMPH"
    )
    x = raw[:, temp_fast_channel]
    x_emph = raw[:, temp_emph_channel]

    raw[:, temp_emph_channel] = deconvolute_mss_ntchp(
        x, x_emph, sampfreq=sampfreq
    )
    data_lists = mx.convert_raw_mrd(raw, probeconf_fname)

    data = {k: np.array(v) for k, v in data_lists.items()}
    data["TEMP"] = fofonoff_filt(data["TEMP"], 55)

    data["SIGMA0"], data["SA"], data["CT"] = calc_ctd(
        data["COND"], data["TEMP"], data["PRESSURE"], lon, lat
    )

    data["pitch"] = data["ACCEL_X"] * 180 / np.pi
    data["roll"] = data["ACCEL_Y"] * 180 / np.pi

    return data


def mss_shear_physical(
    vsink: Float[ndarray, "time"],
    shear_channels: Float[ndarray, "n_shear time"],
    sampfreq=1024.0,
) -> Float[ndarray, "n_shear time"]:
    """Calculates physical shear for the MSS shear probes.
    Returns:
        shear channels in physical units, in same order as passed in

    TODO: data needs canonical column names
    """
    target_freq = 256.0

    # convert from MSS "shear" (i.e. a velocity) to time derivative (units m/s/s)
    shear_channels_phys = []
    for i, sh in enumerate(shear_channels):
        sh_phys = fft_grad(sh, 1 / sampfreq) / vsink**2 / 1025
        shear_channels_phys.append(sh_phys)

    return np.array(shear_channels_phys)
