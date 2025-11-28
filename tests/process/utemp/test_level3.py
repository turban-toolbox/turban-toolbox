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
from turban.process.temperature.level3 import get_noise
from turban.process.temperature.api import TempLevel1, TempConfig, TempProcessing
from turban.utils.util import define_sections
from turban.instruments.mss.mss_utils import deconvolve_mss_ntchp

top_level = Path(__file__).resolve().parent.parent.parent.parent

def test_get_noise():
    x = np.arange(24, dtype=float).reshape(2, 3, 4)
    y = get_noise(x)
    assert y.shape == (2, 4)