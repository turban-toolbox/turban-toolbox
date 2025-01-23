"""
Test the entire processing pipeline
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from atomixpy.level1 import get_vsink
from atomixpy.level2 import select_sections
from atomixpy.process import microtemp, shear
from atomixpy.mss import convert_mrd_to_parquet, level1

os.environ["RUST_BACKTRACE"] = "1"

# raw, _ = convert_mrd_to_parquet(
#     "/home/doppler/data/MSS/youngsound2015/raw/CAST1755.MRD",
# )

raw = pd.read_parquet('CAST1755.pq').values

data = level1(
    raw,
    probeconf_fname="probeconf_mss053_youngsound.json",
    lon=-20.0,
    lat=70.0,
    sampling_freq=1024.0,
)

data["pspd"], data["pressure_lp"] = get_vsink(data["PRESSURE"], 1024.0)
pspda = np.nanmedian(data["pspd"])


def _load_0030():
    ds = xr.open_dataset(
        "/home/doppler/instruments/MSS/data/cast0030.nc", group="L2_cleaned"
    ).isel(TIME=slice(30000, 200_000))
    pspd = ds.PSPD_REL.values
    secno = ds.SECTION_NUMBER
    ds = xr.open_dataset(
        "/home/doppler/instruments/MSS/data/cast0030.nc", group="L1_converted"
    ).isel(TIME=slice(30000, 200_000))
    dTdt = ds.GRADT.isel(N_GRADT_SENSORS=0).values * pspd
    temp = ds.TEMP.isel(N_TEMP_SENSORS=0).values


def test_temp():
    sampling_freq = 1024.0
    pspd, pressure_lp = get_vsink(data["PRESSURE"], sampling_freq)
    pspda = np.nanmedian(pspd)
    section_select_criteria = [
        ((5.0, None), pressure_lp),
        ((0.9 * pspda, 1.1 * pspda), pspd),
    ]

    section_select_idx = select_sections(section_select_criteria)

    microtemp(
        data["TEMP_EMPH"],
        pspd,
        section_select_idx,
        sampling_freq=sampling_freq,
        fftlen=2048,
        chunklen=5,
        chunkoverlap=2,
    )


def test_shear():

    section_select_criteria = [
        ((5.0, None), data["pressure_lp"]),
        ((0.9 * pspda, 1.1 * pspda), data["pspd"]),
        ((-20.0, 20.0), data["pitch"]),
        ((-20.0, 20.0), data["roll"]),
    ]

    section_select_idx = select_sections(
        section_select_criteria, segment_min_len=2048 * 3
    )

    wavenumber, spectra, epsilon = shear(
        np.array([data[k] for k in ["SHEAR_1", "SHEAR_2"]]),
        data["pspd"],
        section_select_idx,
        fftlen=2048,
        sampling_freq=1024.0,
        chunklen=5,
        chunkoverlap=2,
    )


def test_atomix_benchmark():
    import atomixpy.atomixrs as mx
    import xarray as xr

    import numpy as np
    import pandas as pd

    ds = xr.load_dataset("MSS_BalticSea/MSS_Baltic.nc", group="L2_cleaned")

    dt = ds.TIME.diff("TIME").mean().values / pd.Timedelta(seconds=1)

    fft_length = 2048
    diss_length = 5120
    offset = 10000
    x = ds.SHEAR.isel(
        N_SHEAR_SENSORS=0, TIME=slice(offset, offset + diss_length)
    ).values
    pspd = ds.PSPD_REL.isel(TIME_SLOW=slice(offset, offset + diss_length)).values.mean()

    psi, k = np.array(
        mx.process_level3(x, pspd, fft_length, 1 / dt, k0=50, freq_highpass=0.15)
    )

    eps = mx.process_level4(psi, wavenumber=k, platform_speed=pspd)
    print(f"eps is : {eps:2.2e}")
