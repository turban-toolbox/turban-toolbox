import matplotlib.pyplot as plt

from turban.utils.plot.generic import plot_section_numbers, plot_quality_metric
from turban.process.shear.level4 import QUALITY_METRIC_CODES
from turban.process.shear.api import (
    ShearProcessing,
    ShearLevel1,
    ShearLevel2,
    ShearLevel3,
    ShearLevel4,
)


def plot_level1(data: ShearLevel1):
    """Plot Level 1 data with shear and senspeed in two panels."""
    ds = data.to_xarray()
    data_vars = set(ds.data_vars) - {"section_number", "shear", "senspeed"}
    n_panels = len(data_vars)

    fig, axs = plt.subplots(n_panels + 2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Shear data
    ax = axs[0]
    for i in ds["nshear"]:
        ds.isel(nshear=i).shear.plot(ax=ax, x="time")
    ax.set_ylabel("Shear")
    ax.set_xlabel("")
    ax.legend([f"Shear {i+1:1d}" for i in ds["nshear"]])

    # Panel 2: Senspeed
    ax = axs[1]
    ds["senspeed"].plot(ax=ax, x="time", color="k", linewidth=1)
    ax.set_ylabel("Senspeed")
    ax.set_xlabel("Time")

    # Remaining panels
    for ax, dvar in zip(axs[2:], data_vars):
        ds[dvar].plot(ax=ax, x="time", color="k", linewidth=1)

    for ax in axs:
        ax.grid(True, alpha=0.3)

    plot_section_numbers(list(axs), ds.time.values, ds.section_number.values)

    plt.tight_layout()
    return fig, axs


def plot_level2(data: ShearLevel2, data_l1: ShearLevel1 | None = None):
    """Plot Level 1 data with shear and num_despike_iter in two panels."""
    ds = data.to_xarray()
    if data_l1 is not None:
        ds1 = data_l1.to_xarray()
    nshear = len(ds.nshear)

    fig, axs = plt.subplots(nshear * 2, 1, figsize=(12, 4 + nshear * 2), sharex=True)

    for i, (ax1, ax2) in enumerate(zip(axs[::2], axs[1::2])):
        # Panel 1: Shear data
        ax = ax1
        if data_l1 is not None:
            ds1.isel(nshear=i).shear.plot(ax=ax, x="time", c="k")
        ds.isel(nshear=i).shear.plot(ax=ax, x="time", c="r")
        ax.set_title(f"Shear {i+1:1d}")

        # Panel 2:
        ax = ax2
        ds["num_despike_iter"].isel(nshear=i).plot(
            ax=ax, x="time", color="k", linewidth=1
        )
        ax.set_xlabel("Time")
        ax.set_title(f"Shear {i+1:1d}")

    for ax in axs:
        ax.grid(True, alpha=0.3)

    plot_section_numbers(axs, ds.time.values, ds.section_number.values)

    plt.tight_layout()
    return fig, axs


def plot_level3(data: ShearLevel3):
    """Plot shear spectra for each sensor"""
    ds = data.to_xarray()
    nshear = len(ds.nshear)

    fig, axs = plt.subplots(nshear, 1, figsize=(8, 2 + 2 * nshear), sharex=True)
    if nshear == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        # Plot Pk vs wavenumber with time overlaid
        ds_sensor = ds.isel(nshear=i)
        ds_sensor["psi_k_sh"].plot(ax=ax, x="waveno", hue="time", add_legend=False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Shear {i+1}")
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig, axs


def plot_level4(data: ShearLevel4):
    """Plot eps time series and quality metrics for each sensor"""
    ds = data.to_xarray()
    nshear = len(ds.nshear)

    # Create figure with panels for eps and quality metrics
    fig, axs = plt.subplots(nshear + 1, 1, figsize=(12, 3 + 2 * nshear), sharex=True)
    if nshear == 0:
        axs = [axs]

    # Panel 1: Eps time series for all sensors
    ax = axs[0]
    for i in range(nshear):
        ds.eps.isel(nshear=i).plot(ax=ax, x="time", label=f"Shear {i+1}")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Turbulent Dissipation Rate")

    # Remaining panels: Quality metrics for each sensor
    for i in range(nshear):
        ax = axs[i + 1]
        plot_quality_metric(
            ax,
            ds.time.values,
            ds.quality_metric.isel(nshear=i).values,
            q_codes=QUALITY_METRIC_CODES,
            maxq=16,
        )
        ax.set_title(f"Shear {i+1} Quality Metrics")

    plt.tight_layout()
    return fig, axs
