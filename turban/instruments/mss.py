import os
from pathlib import Path
from beartype.typing import Dict

import numpy as np
from jaxtyping import Num
from numpy import ndarray
import pandas as pd

from pydantic import BaseModel

from turban.ctd import *
from turban.shear.level1 import *
from turban.temperature.temperature import *
from turban.util import channel_mapping

from turban.instruments.instrument import Dropsonde
from turban.shear import ShearLevel1
from turban.util import get_vsink

class MSS(Dropsonde):

    def read_mrd(self, fname: str):
        raise NotImplementedError

    def to_level1(self, pressure_raw, shear_raw, cfg: ShearConfig):
        pspd, pressure_lp = get_vsink(pressure_raw, cfg.sampling_freq)
        shear_phys = mss_shear_physical(pspd, shear_raw, self.sampling_freq)
        return ShearLevel1(
            pspd=pspd,
            shear_phys=shear_phys,
            cfg=cfg,
        )


def convert_mrd_to_parquet(
    mrd_fname: str,
    parquet_fname: str | None = None,
) -> Tuple[Int[ndarray, "time 16"], str]:
    if parquet_fname is None:
        parquet_fname = Path(mrd_fname).with_suffix(".pq")

    raw = mx.read_mrd(mrd_fname)

    raw = np.array(raw).copy()
    pd.DataFrame(raw, columns=[f"{i:02d}" for i in range(1, 17)]).to_parquet(
        parquet_fname
    )

    return raw, str(parquet_fname)


def level1(
    raw: Int[ndarray, "time 16"],
    probeconf_fname: str,
    lon: float,
    lat: float,
    sampling_freq: float = 1024.0,
) -> Dict[str, Num[ndarray, "time"]]:
    """
    Convert raw MSS data to physical units
    """
    temp_fast_channel, temp_emph_channel = channel_mapping(
        probeconf_fname, "TEMP_FAST", "TEMP_EMPH"
    )
    x = raw[:, temp_fast_channel]
    x_emph = raw[:, temp_emph_channel]

    raw[:, temp_emph_channel] = deconvolute_mss_ntchp(
        x, x_emph, sampling_freq=sampling_freq
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
    sampling_freq=1024.0,
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
        sh_phys = fft_grad(sh, 1 / sampling_freq) / vsink**2 / sea_water_density
        shear_channels_phys.append(sh_phys)

    return np.array(shear_channels_phys)
