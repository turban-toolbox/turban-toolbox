import os
from pathlib import Path
from beartype.typing import Dict

import numpy as np
from jaxtyping import Num
from numpy import ndarray
import pandas as pd

import atomixpy.atomixrs as mx
from atomixpy.ctd import *
from atomixpy.level1 import *
from atomixpy.temperature import *
from atomixpy.util import channel_mapping

os.environ["RUST_BACKTRACE"] = "1"


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
        x, x_emph, sampling_freq_Hz=sampling_freq
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
