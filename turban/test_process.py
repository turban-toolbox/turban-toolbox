"""
Test the entire processing pipeline
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from turban.level1 import get_vsink
from turban.level2 import select_sections
from turban.process import microtemp, shear
from turban.mss import convert_mrd_to_parquet, level1

os.environ["RUST_BACKTRACE"] = "1"

# raw, _ = convert_mrd_to_parquet(
#     "/home/doppler/data/MSS/youngsound2015/raw/CAST1755.MRD",
# )

# raw = pd.read_parquet("CAST1755.pq").values

# data = level1(
#     raw,
#     probeconf_fname="probeconf_mss053_2024.json",
#     lon=-20.0,
#     lat=70.0,
#     sampling_freq=1024.0,
# )

# data["pspd"], data["pressure_lp"] = get_vsink(data["PRESSURE"], 1024.0)
# pspda = np.nanmedian(data["pspd"])


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


def plot_spectra(ds, canvas_kwarg, shade_kwarg):

    import datashader as dsh

    level3 = ds.copy()
    level3["k"].loc[
        {
            "wavenumber": 0,
        }
    ] = np.nan
    level3["Pk"].loc[
        {
            "wavenumber": 0,
        }
    ] = np.nan
    df = level3.to_dataframe()
    cvs = dsh.Canvas(**canvas_kwarg)

    agg = cvs.line(source=df, x="k", y="Pk", agg=dsh.count())
    im = dsh.transfer_functions.shade(agg, how="eq_hist", **shade_kwarg)

    return im


def test_baltic_benchmark():
    import xarray as xr

    import numpy as np
    import pandas as pd

    from turban.level2 import process_level2
    from turban.level3 import process_level3
    from turban.level4 import process_level4

    ds1 = xr.load_dataset("MSS_BalticSea/MSS_Baltic.nc", group="L1_converted")
    ds2 = xr.load_dataset("MSS_BalticSea/MSS_Baltic.nc", group="L2_cleaned")
    ds3 = xr.load_dataset("MSS_BalticSea/MSS_Baltic.nc", group="L3_spectra")
    ds4 = xr.load_dataset("MSS_BalticSea/MSS_Baltic.nc", group="L4_dissipation")

    (idx,) = np.where(ds2.SECTION_NUMBER == 1)
    level2 = process_level2(
        shear=ds1.SHEAR.values,
        section_select_idx=[idx.tolist()],
        sampling_freq_Hz=1024.0,
        fftlen=2048,
    )

    assert len(level2) == 1
    shear_cleaned, is_despiked, n_iter = level2[0]

    dt = ds2.TIME.diff("TIME").mean().values / pd.Timedelta(seconds=1)

    fft_length = 2048

    level3 = process_level3(
        shear_segment=ds2.SHEAR.values,
        pspd_segment=ds2.PSPD_REL.values,
        fftlen=fft_length,
        sampling_freq=1 / dt,
        spatial_response_wavenum=50.0,
        freq_highpass=0.15,
        chunklen=5,
        chunkoverlap=2,
    )

    level4 = process_level4(
        level3.Pk.values, level3.k.values, level3.platform_speed.values
    )

    # comparison plots
    import datashader as dsh

    canvas_kwarg = dict(
        plot_height=500,
        plot_width=500,
        x_range=[10**0, 10**3],
        y_range=[1e-10, 1e-1],
        x_axis_type="log",
        y_axis_type="log",
    )
    xticks, xticklabels = zip(
        *[
            (
                canvas_kwarg["plot_width"]
                / (
                    np.log10(canvas_kwarg["x_range"][1])
                    - np.log10(canvas_kwarg["x_range"][0])
                )
                * l,
                10**l,
            )
            for l in np.arange(0, 3.1)
        ]
    )
    yticks, yticklabels = zip(
        *[
            (
                canvas_kwarg["plot_height"]
                / (
                    np.log10(canvas_kwarg["y_range"][1])
                    - np.log10(canvas_kwarg["y_range"][0])
                )
                * (np.log10(canvas_kwarg["y_range"][1]) - l),
                10.0**l,
            )
            for l in np.arange(-10, -0.9)
        ]
    )

    for nshear in [0, 1]:
        im = dsh.transfer_functions.stack(
            plot_spectra(
                ds3.rename(
                    {
                        "N_SHEAR_SENSORS": "nshear",
                        "SH_SPEC": "Pk",
                        "KCYC": "k",
                        "WAVENUMBER": "wavenumber",
                    }
                ).isel(nshear=nshear),
                canvas_kwarg=canvas_kwarg,
                shade_kwarg=dict(cmap="black"),
            ),
            plot_spectra(
                level3.isel(nshear=nshear),
                canvas_kwarg=canvas_kwarg,
                shade_kwarg=dict(cmap="red"),
            ),
        )
        fig = plt.figure(figsize=(9, 9))
        ax = fig.subplots()
        ax.imshow(im.to_pil())
        ax.set_xticks(xticks, xticklabels)
        ax.set_yticks(yticks, yticklabels)
        ax.set_xlabel("wavenumber")
        ax.set_ylabel("Power spectral density (s^-2) / (m^-1)")
        fig.savefig(f"baltic-level3-shear-{nshear}.png")
