import os

os.environ["RUST_BACKTRACE"] = "1"

from jaxtyping import Num
from beartype.typing import Dict
import pandas as pd


from turban.ctd import *
from turban.shear.level1 import *
from turban.temperature.temperature import *
from numpy import ndarray
from turban.mss import convert_mrd_to_parquet


def test_convert_mrd():
    raw, parquet_fname = convert_mrd_to_parquet(
        "/home/doppler/data/MSS/youngsound2015/raw/CAST1755.MRD",
        parquet_fname="CAST1755.pq",
    )
    assert parquet_fname == "CAST1755.pq"
    raw2 = pd.read_parquet(parquet_fname)
    assert np.allclose(raw, raw2)
