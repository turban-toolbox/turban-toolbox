#!/bin/env python
import os

os.environ["RUST_BACKTRACE"] = "1"
import atomixpy.atomixrs as mx

import xarray as xr

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

ds = xr.load_dataset("MSS_BalticSea/MSS_Baltic.nc", group="L2_cleaned")

dt = ds.TIME.diff("TIME").mean().values / pd.Timedelta(seconds=1)

fft_length = 2048
diss_length = 5120
offset = 10000
x = ds.SHEAR.isel(
    N_SHEAR_SENSORS=0, TIME=slice(offset, offset + diss_length)
).values
pspd = ds.PSPD_REL.isel(TIME_SLOW=slice(offset, offset + diss_length)).values.mean()

xx = ds.SHEAR.isel(N_SHEAR_SENSORS=0).values

psi, k = np.array(
    mx.process_level3(x, pspd, fft_length, 1 / dt, k0=50, freq_highpass=0.15)
)
# mx.process_all(ds.SHEAR.isel(N_SHEAR_SENSORS=0).values, ds.PSPD_REL.values, diss_length=diss_length, fft_length=fft_length, diss_length_overlap=0, sampling_freq=1/dt)

eps = mx.process_level4(psi, wavenumber=k, platform_speed=pspd)
print(f"eps is : {eps:2.2e}")
