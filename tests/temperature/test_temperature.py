import pytest
import numpy as np

from turban.temperature import * # TODO use named imports
from turban.temperature.temperature import microtemp


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

