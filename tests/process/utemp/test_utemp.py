from pathlib import Path
import json
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from turban.instruments.mss import mss_mrd
from turban.instruments.mss.api import mrd_to_level1
from turban.instruments.mss.config import MssDeviceConfig
from turban.instruments.mss.mss_mrd import read_mrd, level0_to_level1, raw_to_level0
from turban.process.utemp.level3 import get_noise
from turban.process.utemp.api import UTempConfig, UTempProcessing
from turban.utils.util import define_sections
from turban.instruments.mss.mss_utils import deconvolve_mss_ntchp

from turban.utils.filepaths import mss_probeconf_json_fpath, mss_utemp_mrd_fpath


def test_mss():

    with open(mss_probeconf_json_fpath) as f:
        mss_cfg_053 = MssDeviceConfig.model_validate(json.load(f))

    cfg = UTempConfig(
        sampfreq=1024.0,
        segment_length=2048,
        segment_overlap=1024,
        chunk_length=4 * 2048,
        chunk_overlap=1024,
        waveno_limit_upper=500.0,
        diff_gain=1.5,
    )
    ut1 = mrd_to_level1(mss_utemp_mrd_fpath, "utemp", cfg, mss_cfg_053)
    p = UTempProcessing(ut1)
    ds4 = p.level4.to_xarray()
