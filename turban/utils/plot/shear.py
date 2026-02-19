import matplotlib.pyplot as plt
import xarray as xr
from typing import Any, cast

from turban.utils.plot.generic import plot_section_numbers, plot_quality_metric
from turban.process.shear.level4 import QUALITY_METRIC_CODES
from turban.process.shear.api import (
    ShearProcessing,
    ShearLevel1,
    ShearLevel2,
    ShearLevel3,
    ShearLevel4,
)

ShearLevelType = ShearLevel1 | ShearLevel2 | ShearLevel3 | ShearLevel4


def _to_levels(data: Any) -> tuple[ShearLevelType, ...]:
    if isinstance(data, ShearProcessing):
        out = tuple(
            cast(ShearLevelType, level)
            for level in (data.level1, data.level2, data.level3, data.level4)
            if isinstance(level, (ShearLevel1, ShearLevel2, ShearLevel3, ShearLevel4))
        )
        if len(out) == 0:
            raise ValueError("ShearProcessing does not contain any levels")

    elif isinstance(data, (ShearLevel1, ShearLevel2, ShearLevel3, ShearLevel4)):
        out = (data,)

    elif isinstance(data, xr.Dataset):
        out = []
        for class_ in (ShearLevel1, ShearLevel2, ShearLevel3, ShearLevel4):
            try:
                out.append(class_.from_xarray(data))
            except Exception:
                continue
        if len(out) == 0:
            raise ValueError(
                "Could not convert dataset to ShearLevel1-4 via from_xarray"
            )
        out = tuple(out)

    elif isinstance(data, xr.DataTree):
        out = []
        for level, class_ in (
            (1, ShearLevel1),
            (2, ShearLevel2),
            (3, ShearLevel3),
            (4, ShearLevel4),
        ):
            level_name = f"level{level}"
            if level_name in data:
                ds = data[level_name].to_dataset()
                out.append(class_.from_xarray(ds))
        if len(out) == 0:
            raise ValueError("Could not find level1-4 in DataTree")
        out = tuple(out)

    else:
        raise TypeError(
            "Input must be ShearProcessing, xarray.Dataset, xarray.DataTree, or ShearLevel1-4 instance"
        )

    print(f"Mapped {type(data).__name__} to {tuple(type(o).__name__ for  o in out)}")

    return out


def plot(*data: Any):
    plot_map = {
        1: plot_level1,
        2: plot_level2,
        3: plot_level3,
        4: plot_level4,
    }

    level_data_items = [level for item in data for level in _to_levels(item)]
    level1_ref = next((item for item in level_data_items if item._level == 1), None)

    out = []
    for level_data in level_data_items:
        if level_data._level == 2 and level1_ref is not None:
            out.append(plot_level2(level_data, level1_ref))
        else:
            out.append(plot_map[level_data._level](level_data))

    return out


def _to_dataset(data: ShearLevelType | xr.Dataset) -> xr.Dataset:
    if isinstance(data, xr.Dataset):
        return data
    return data.to_xarray()


def plot_level1(data: ShearLevel1 | xr.Dataset):
    """Plot Level 1 data with shear and senspeed in two panels."""
    ds = _to_dataset(data)
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

    fig.suptitle("Level 1")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axs


def plot_level2(
    data: ShearLevel2 | xr.Dataset, data_l1: ShearLevel1 | xr.Dataset | None = None
):
    """Plot Level 2 data. If data_l1 is given, plots uncleaned shear for comparison."""
    ds = _to_dataset(data)
    valid_section = ds["section_number"] != 0
    if data_l1 is not None:
        ds1 = _to_dataset(data_l1)
        valid_section_l1 = ds1["section_number"] != 0
    nshear = len(ds.nshear)

    fig, axs = plt.subplots(nshear * 2, 1, figsize=(12, 4 + nshear * 2), sharex=True)

    for i, (ax1, ax2) in enumerate(zip(axs[::2], axs[1::2])):
        # Panel 1: Shear data
        ax = ax1
        if data_l1 is not None:
            ds1.isel(nshear=i).shear.where(valid_section_l1).plot(
                ax=ax, x="time", c="k"
            )
        ds.isel(nshear=i).shear.where(valid_section).plot(ax=ax, x="time", c="r")
        ax.set_title(f"Shear {i+1:1d}")

        # Panel 2:
        ax = ax2
        ds["num_despike_iter"].isel(nshear=i).where(valid_section).plot(
            ax=ax, x="time", color="k", linewidth=1
        )
        ax.set_xlabel("Time")
        ax.set_title(f"Shear {i+1:1d}")

    for ax in axs:
        ax.grid(True, alpha=0.3)

    plot_section_numbers(axs, ds.time.values, ds.section_number.values)

    fig.suptitle("Level 2")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axs


def plot_level3(data: ShearLevel3 | xr.Dataset):
    """Plot shear spectra for each sensor"""
    ds = _to_dataset(data)
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

    fig.suptitle("Level 3")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axs


def plot_level4(data: ShearLevel4 | xr.Dataset):
    """Plot eps time series and quality metrics for each sensor"""
    ds = _to_dataset(data)
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

    fig.suptitle("Level 4")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axs
