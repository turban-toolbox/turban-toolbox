from pytest import fixture
import numpy as np
from turban.shear.level2 import select_sections
import xarray as xr

get_vsink = data = None
from turban.temperature import *  # TODO use named imports
from turban.temperature.temperature import microtemp


@fixture
def microtemp_mss_data():
    fname = "/home/doppler/instruments/MSS/data/cast0030.nc"
    ds = xr.open_dataset(fname, group="L2_cleaned").isel(TIME=slice(30000, 200_000))
    pspd = ds.PSPD_REL.values
    secno = ds.SECTION_NUMBER
    ds = xr.open_dataset(fname, group="L1_converted").isel(TIME=slice(30000, 200_000))
    dTdt = ds.GRADT.isel(N_GRADT_SENSORS=0).values * pspd
    temp = ds.TEMP.isel(N_TEMP_SENSORS=0).values
    return dTdt, pspd, secno, temp


def _test_temp():
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
        fft_length=2048,
        chunklen=5,
        chunkoverlap=2,
    )
