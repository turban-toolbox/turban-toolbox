from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _plotall(data, outpath):
    Path(outpath).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    for v in df:
        plt.figure()
        df[v].plot().get_figure().savefig(f"{outpath}/{v}.png")


import panel as pn
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import hvplot.xarray


def plot_df_col_mpl(df, v):
    fig = df[v].plot(ylabel=v).get_figure()
    plt.close(fig)
    return fig


def plot_spec_mpl(ds3, i):
    k = ds3["k"].values
    Pk = ds3["Pk"].values[i]

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    ax.loglog(k.transpose(), Pk.transpose())
    ax.set_xlabel("wavenumber ()")
    ax.set_title(f"Shear microstructure (sensor #{i})")
    # xlim=(0.1, None),
    # ylim=(1e-11, None),
    ax.set_ylabel(f"Shear variance (1 s-2 s-1) (sensor #{i})")
    return fig


def _plot_spec_hv():
    ds3[["k", "Pk"]].isel(nshear=i).hvplot(
        x="k",
        y="Pk",
        by="time_slow",
        logx=True,
        logy=True,
        title=f"Shear microstructure (sensor #{i})",
        xlim=(0.1, None),
        ylim=(1e-11, None),
        xlabel="wavenumber ()",
        ylabel=f"Shear variance (1 s-2 s-1) (sensor #{i})",
    )


def plot_microtemp_spec_binavg(dst):
    df = dst.to_dataframe()

    kbin = np.linspace(0.01, 3e3, 1000)
    klabel = kbin[:-1] + np.diff(kbin) / 2
    chibin = 10.0 ** np.arange(-15, -5, 1 / 2)
    chilabel = chibin[:-1]

    df["kbin"] = pd.cut(df["k"], bins=kbin, labels=klabel)
    df["chibin"] = pd.cut(df["chi"], bins=chibin, labels=chilabel)

    ds0 = df.groupby(["kbin", "chibin"])[["k", "Pk"]].mean().dropna().to_xarray()

    ds = ds0

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    ax.loglog(ds["k"].values, ds["Pk"].values)
    ax.legend(ds.chibin.values)
    ax.set_ylim(1e-11, 1e-4)
    return fig


def dashboard(fname_data: str, fname_out: str | None = None):
    with Dataset(fname_data) as f:
        groups = list(f.groups)

    if "level0" in groups:
        ds0 = xr.load_dataset(fname_data, group="level0")
        df0 = ds0.to_dataframe()
        raw = pn.Accordion(
            *[(v, pn.pane.Matplotlib(plot_df_col_mpl(df0, v))) for v in sorted(df0)]
        )
    else:
        raw = None

    if "microtemp" in groups:
        dst = xr.load_dataset(fname_data, group="microtemp")

        spec_microtemp = pn.Row(
            dst.hvplot(
                x="k",
                y="Pk",
                by="time_slow",
                logx=True,
                logy=True,
                title="Temperature microstructure",
                xlim=(0.1, None),
                ylim=(1e-10, None),
                xlabel="wavenumber ()",
                ylabel="Temperature gradient variance (K2 m-2 s-1)",
            ),
            plot_microtemp_spec_binavg(dst),
        )
    else:
        spec_microtemp = None

    if "level3" in groups:
        ds3 = xr.load_dataset(fname_data, group="level3")
        spec_shear = pn.Column(*[plot_spec_mpl(ds3, i) for i in range(len(ds3.nshear))])
    else:
        spec_shear = None

    l = pn.Tabs(
        ("Raw data", raw),
        ("Spectra", pn.Column(spec_microtemp, spec_shear)),
    )

    if fname_out is not None:
        from bokeh.resources import INLINE

        l.save(fname_out, resources=INLINE)

    return l
