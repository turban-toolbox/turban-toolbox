import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Any, cast, TypeAlias
import logging

from turban.utils.plot.generic import plot_section_numbers, plot_quality_metric
from turban.process.shear.level4 import QUALITY_METRIC_CODES
from turban.process.shear.api import (
    ShearProcessing,
    ShearLevel1,
    ShearLevel2,
    ShearLevel3,
    ShearLevel4,
)
from turban.process.shear.util import model_spectrum
from turban.utils.util import define_sections

ShearLevelType = ShearLevel1 | ShearLevel2 | ShearLevel3 | ShearLevel4
SubsetSpec: TypeAlias = list[tuple[str, Any, Any]]

logger = logging.getLogger(__name__)


def _to_levels(data: Any) -> tuple[ShearLevelType, ...]:
    """Convert a data object to a tuple of ShearLevel instances.

    Parameters
    ----------
    data : Any
        Input data: ``ShearProcessing``, ``ShearLevel1``–``ShearLevel4``,
        ``xr.Dataset``, or ``xr.DataTree``.

    Returns
    -------
    tuple of ShearLevelType
        One or more shear level instances extracted from ``data``.

    Raises
    ------
    TypeError
        If ``data`` is not a recognised type.
    ValueError
        If no valid levels can be extracted from ``data``.
    """
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

    logger.info(
        f"Mapped {type(data).__name__} to {tuple(type(o).__name__ for  o in out)}"
    )

    return out


def plot(*data: Any, subset: SubsetSpec | None = None):
    """Make all possible plots from any number of supplied data."""
    plot_map = {
        1: plot_level1,
        2: plot_level2,
        3: plot_level3,
        4: plot_level4,
    }
    level_data_items = [level for item in data for level in _to_levels(item)]
    level_data_levels = [item._level for item in level_data_items]

    if len(set(level_data_levels)) < len(level_data_levels):
        raise ValueError("Some levels occur more than once; do not know what to do")

    out = []
    for level, plotfunc in plot_map.items():
        if level in level_data_levels:
            out.append(plotfunc(*level_data_items, subset=subset))

    return out


def _to_dataset(data: ShearLevelType | xr.Dataset) -> xr.Dataset:
    """Convert a ShearLevel instance or Dataset to an xarray Dataset.

    Parameters
    ----------
    data : ShearLevelType or xr.Dataset
        Data to convert.

    Returns
    -------
    xr.Dataset
        The dataset representation of ``data``.
    """
    if isinstance(data, xr.Dataset):
        return data
    return data.to_xarray()


def _parse_level_inputs(*data: Any) -> dict[int, ShearLevelType]:
    """Parse mixed data inputs into a dict keyed by level number.

    Parameters
    ----------
    *data : Any
        Any combination of ``ShearProcessing``, ``ShearLevel1``–``ShearLevel4``,
        ``xr.Dataset``, or ``xr.DataTree`` objects.

    Returns
    -------
    dict[int, ShearLevelType]
        Mapping from level number (1–4) to corresponding ShearLevel instance.
    """
    level_items = tuple(level for item in data for level in _to_levels(item))
    out = {}
    for i in level_items:
        out[i._level] = i
    return out


def _clip(ds: xr.Dataset, subset: SubsetSpec | None = None):
    """Subset a dataset to time steps that satisfy all subset bounds.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to subset.
    subset : list of tuple, optional
        List of ``(variable, vmin, vmax)`` triples. Only time steps where every
        variable falls within its ``[vmin, vmax]`` range are retained.
        If None or empty, ``ds`` is returned unchanged.

    Returns
    -------
    xr.Dataset
        Subsetted dataset.
    """
    if subset is None or len(subset) == 0:
        return ds
    data_and_bounds = [(ds[var].values, vmin, vmax) for var, vmin, vmax in subset]
    ds = ds.sel(time=define_sections(*data_and_bounds) > 0)
    return ds


def _subset_suffix(subset: SubsetSpec | None) -> str:
    """Build a human-readable suffix string describing the active subset.

    Parameters
    ----------
    subset : list of tuple or None
        List of ``(variable, vmin, vmax)`` triples, or None.

    Returns
    -------
    str
        Empty string if no subset is active, otherwise a newline-prefixed
        description of all subset bounds.
    """
    if subset is None or len(subset) == 0:
        return ""
    subset_str = "; ".join(f"{var}∈[{vmin}, {vmax}]" for var, vmin, vmax in subset)
    return f"\nsubset: {subset_str}"


def plot_level1(*data: Any, subset: SubsetSpec | None = None):
    """Plot Level 1 data with shear and senspeed panels.

    Parameters
    ----------
    *data : Any
        ShearLevel1, ShearProcessing, xr.Dataset, or xr.DataTree objects.
    subset : list of tuple, optional
        List of ``(variable, vmin, vmax)`` bounds for subsetting. If None, no
        subsetting is applied.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, ndarray of Axes)
        Figure with panels for shear (one line per sensor), senspeed, and any
        remaining data variables.
    """
    levels = _parse_level_inputs(*data)

    ds = _clip(_to_dataset(levels.get(1)), subset)

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

    fig.suptitle(f"Level 1{_subset_suffix(subset)}")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axs


def plot_level2(*data: Any, subset: SubsetSpec | None = None):
    """Plot Level 2 data with cleaned shear and despike iteration counts.

    Parameters
    ----------
    *data : Any
        ShearLevel2 required; ShearLevel1 optional (for overlay comparison).
        Also accepts ShearProcessing, xr.Dataset, or xr.DataTree objects.
    subset : list of tuple, optional
        List of ``(variable, vmin, vmax)`` bounds for subsetting. If None, no
        subsetting is applied.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, ndarray of Axes)
        Figure with 2×nshear panels: cleaned vs raw shear overlay per sensor,
        and despike iteration count per sensor.
    """
    levels = _parse_level_inputs(*data)

    ds = _clip(_to_dataset(levels.get(2)), subset)
    valid_section = ds["section_number"] != 0
    data_l1 = levels.get(1, None)
    ds1 = None
    if data_l1 is not None:
        ds1 = _clip(_to_dataset(data_l1), subset)
        valid_section_l1 = ds1["section_number"] != 0
    nshear = len(ds.nshear)

    fig, axs = plt.subplots(nshear * 2, 1, figsize=(12, 4 + nshear * 2), sharex=True)

    for i, (ax1, ax2) in enumerate(zip(axs[::2], axs[1::2])):
        # Panel 1: Shear data
        ax = ax1
        if ds1 is not None:
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

    fig.suptitle(f"Level 2{_subset_suffix(subset)}")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axs


def plot_level3(*data: Any, subset: SubsetSpec | None = None):
    """Plot shear spectra and time series with optional model spectrum overlay.

    Parameters
    ----------
    *data : Any
        ShearLevel3 required; ShearLevel4 optional (for model spectrum overlay).
        Also accepts ShearProcessing, xr.Dataset, or xr.DataTree objects.
    subset : list of tuple, optional
        List of ``(variable, vmin, vmax)`` bounds for subsetting. If None, no
        subsetting is applied.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, list of Axes)
        Figure with shear spectra panels (one per sensor) and time series panels
        for senspeed and auxiliary variables. Model Nasmyth spectra are overlaid
        if Level 4 data is provided.
    """
    levels = _parse_level_inputs(*data)
    data_l3 = levels.get(3)
    data_l4 = levels.get(4, None)

    if data_l3 is None:
        raise ValueError("plot_level3 requires Level 3 data")

    ds3 = _clip(_to_dataset(data_l3), subset)
    ds4 = _clip(_to_dataset(data_l4), subset) if data_l4 is not None else None
    nshear = len(ds3.nshear)

    aux_keys = set((getattr(data_l3, "_aux_data", {}) or {}).keys())
    aux_vars = [
        var for var in ds3.data_vars if var in aux_keys and "time" in ds3[var].dims
    ]
    ts_vars = ["senspeed", *aux_vars]

    n_ts = len(ts_vars)
    fig = plt.figure(figsize=(4 * nshear, 3 + 2.2 * (1 + n_ts)))
    gs = fig.add_gridspec(nrows=1 + n_ts, ncols=nshear)

    spectra_axes = [fig.add_subplot(gs[0, i]) for i in range(nshear)]
    ts_axes = [fig.add_subplot(gs[1 + i, :]) for i in range(n_ts)]

    for i, ax in enumerate(spectra_axes):
        # Plot Pk vs wavenumber with time overlaid
        ds_sensor = ds3.isel(nshear=i)
        ds_sensor["psi_k_sh"].plot(ax=ax, x="waveno", hue="time", add_legend=False)

        if ds4 is not None:
            waveno_sensor = np.asarray(ds_sensor["waveno"].values)
            waveno_pos = waveno_sensor[np.isfinite(waveno_sensor) & (waveno_sensor > 0)]
            if waveno_pos.size > 0:
                waveno = np.logspace(
                    np.log10(np.nanmin(waveno_pos)),
                    np.log10(np.nanmax(waveno_pos)),
                    200,
                )
                eps = ds4["eps"].isel(nshear=i).values
                kolm_length = ds4["kolm_length"].isel(nshear=i).values
                molvisc = (eps * kolm_length**4) ** (1 / 3)
                psi = model_spectrum(waveno, eps, molvisc)

                ax.plot(waveno, psi.T, color="k", alpha=0.3, linewidth=2, ls="-")

        ax.set_xscale("log")
        ax.set_ylim(np.nanpercentile(ds3["psi_k_sh"], 0.1), None)
        ax.set_yscale("log")
        ax.set_title(f"Shear {i+1}")
        ax.grid(True, alpha=0.3, which="both")

    # add sensor speed and auxiliary variables
    for ax, var in zip(ts_axes, ts_vars):
        ds3[var].plot(ax=ax, color="k", linewidth=1)
        ax.grid(True, alpha=0.3)

    ts_axes[-1].set_xlabel("Time")
    plot_section_numbers(ts_axes, ds3.time.values, ds3.section_number.values)

    axs = [*spectra_axes, *ts_axes]

    fig.suptitle(f"Level 3{_subset_suffix(subset)}")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axs


def plot_level4(*data: Any, subset: SubsetSpec | None = None):
    """Plot turbulent dissipation rate time series and quality metrics.

    Parameters
    ----------
    *data : Any
        ShearLevel4 required. Also accepts ShearProcessing, xr.Dataset, or
        xr.DataTree objects.
    subset : list of tuple, optional
        List of ``(variable, vmin, vmax)`` bounds for subsetting. If None, no
        subsetting is applied.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, list of Axes)
        Figure with eps time series panel and quality metric panels for each
        sensor.
    """
    levels = _parse_level_inputs(*data)

    ds = _clip(_to_dataset(levels.get(4)), subset)
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

    fig.suptitle(f"Level 4{_subset_suffix(subset)}")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axs
