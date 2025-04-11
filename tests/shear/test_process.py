"""
Test the entire processing pipeline
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from tests.fixtures import atomix_nc_filename


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


def plot_spectra(datasets: dict, canvas_kwarg, shade_kwarg):

    import datashader as dsh

    df = pd.concat(
        ds.to_dataframe().reset_index().assign(source=source)
        for source, ds in datasets.items()
    )
    df["source"] = df["source"].astype("category")
    df = df.drop(df[df.Pk.isna() & df.k > 0].index)

    cvs = dsh.Canvas(**canvas_kwarg)

    agg = cvs.line(source=df, x="k", y="Pk", agg=dsh.by("source"))
    im = dsh.transfer_functions.shade(agg, how="eq_hist", **shade_kwarg)

    return im


def test_load_atomix_netcdf(atomix_nc_filename):
    from turban.shear import ShearProcessing

    for level in [1, 2, 3]:
        p = ShearProcessing.from_atomix_netcdf(atomix_nc_filename, level=level)
        assert isinstance(p.level4.eps, np.ndarray)


def test_baltic_benchmark(atomix_nc_filename):
    import xarray as xr
    from turban.shear import (
        ShearProcessing,
        ShearLevel1,
        ShearLevel2,
        ShearLevel3,
        ShearLevel4,
    )

    p = ShearProcessing.from_atomix_netcdf(atomix_nc_filename, level=1)

    level1 = p.level1
    level2 = p.level2
    level3 = p.level3
    level4 = p.level4
    assert isinstance(level1, ShearLevel1)
    assert isinstance(level2, ShearLevel2)
    assert isinstance(level3, ShearLevel3)
    assert isinstance(level4, ShearLevel4)

    ds1 = xr.load_dataset(atomix_nc_filename, group="L1_converted")
    ds2 = xr.load_dataset(atomix_nc_filename, group="L2_cleaned")
    ds3 = xr.load_dataset(atomix_nc_filename, group="L3_spectra").rename(
        {
            "N_SHEAR_SENSORS": "nshear",
            "SH_SPEC": "Pk",
            "KCYC": "k",
            "WAVENUMBER": "wavenumber",
        }
    )  # for consistency with turban level 3
    ds4 = xr.load_dataset(atomix_nc_filename, group="L4_dissipation")

    ds3_turban = level3.to_xarray()
    ds4_turban = level4.to_xarray()

    ds3_turban.to_netcdf("out/tests/baltic_level3.nc")
    ds4_turban.to_netcdf("out/tests/baltic_level4.nc")

    # _plot_level3(ds3, ds3_turban) # disable for now
    # _plot_level4(ds4, ds4_turban) # TODO


def _plot_despiking(ds1, level1, ds2, level2):
    plt.plot(level1.shear[0])
    plt.plot(level2.shear[0])

    ds1.SHEAR.isel(N_SHEAR_SENSORS=0).plot()
    ds2.SHEAR.isel(N_SHEAR_SENSORS=0).plot()


def _plot_level3(ds3, level3):
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
    ds3["k"].loc[
        {
            "wavenumber": 0,
        }
    ] = np.nan
    ds3["Pk"].loc[
        {
            "wavenumber": 0,
        }
    ] = np.nan

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
    shade_kwarg = dict(color_key={"benchmark": "violet", "turban": "green"})
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
        im = plot_spectra(
            {
                "benchmark": ds3.isel(nshear=nshear, N_SH_VIB_SPEC=nshear),
                "turban": level3.isel(nshear=nshear),
            },
            canvas_kwarg=canvas_kwarg,
            shade_kwarg=shade_kwarg,
        )
        fig = plt.figure(figsize=(9, 9))
        ax = fig.subplots()
        ax.imshow(im.to_pil())
        ax.set_xticks(xticks, xticklabels)
        ax.set_yticks(yticks, yticklabels)
        ax.set_xlabel("wavenumber")
        ax.set_ylabel("Power spectral density")
        ax.set_title(f"{shade_kwarg['color_key']}")
        fig.savefig(f"out/tests/baltic-level3-shear-{nshear}.png")


def _plot_level4(ds4, level4):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.plot(ds4.PRES, ds4.EPSI_FINAL, "k", label="benchmark", marker="o")
    ax.plot(level4.PRES, level4.eps.mean("nshear"), "r", label="turban", marker="o")
    ax.set_xlabel("Pressure (dbar)")
    ax.set_ylabel("Dissipation rate (W/kg)")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig("out/tests/baltic-level4-eps.png")
