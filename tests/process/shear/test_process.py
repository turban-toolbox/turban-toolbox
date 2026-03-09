"""
Test the entire processing pipeline
"""

import pytest

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from tests.filepaths import atomix_benchmark_baltic_fpath, atomix_benchmark_faroe_fpath

import xarray as xr
from turban.process.shear.api import (
    ShearProcessing,
    ShearLevel1,
    ShearLevel2,
    ShearLevel3,
    ShearLevel4,
    NetcdfReader,
)

from turban.process.shear.api import ShearProcessing


@pytest.mark.parametrize("level", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "fpath", [atomix_benchmark_baltic_fpath, atomix_benchmark_faroe_fpath]
)
def test_load_atomix_netcdf(fpath, level):
    p = ShearProcessing.from_atomix_netcdf(fpath, level=level)
    assert isinstance(p.level4.eps, np.ndarray)


def test_agg_aux():
    """Test equivalence of simplified and advanced API for aggregating aux variables"""

    aux_vars = ["temp"]
    arr = dict(
        zip(
            aux_vars,
            NetcdfReader("atomix").read(atomix_benchmark_baltic_fpath, aux_vars),
        )
    )
    data_aux = {
        "temp": (
            ["time"],
            arr["temp"][0, :],
            {"max": "temp_max"},
        ),
    }

    level1 = ShearLevel1.from_atomix_netcdf(atomix_benchmark_baltic_fpath)
    level1.add_aux_data(arr["temp"][0, :], "temp", "max", "temp_max")
    p_from_level1 = ShearProcessing(level1)

    p_from_atomix = ShearProcessing.from_atomix_netcdf(
        atomix_benchmark_baltic_fpath,
        level=1,
        data_aux=data_aux,
    )

    assert p_from_level1.level4.to_xarray().equals(p_from_atomix.level4.to_xarray())


def test_baltic_benchmark():

    aux_vars = ["time", "press", "temp", "cond"]
    arr = dict(
        zip(
            aux_vars,
            NetcdfReader("atomix").read(atomix_benchmark_baltic_fpath, aux_vars),
        )
    )
    data_aux = {
        "time": (
            ["time"],
            arr["time"],
            {"mean": "time_slow"},
        ),
        "temp": (
            ["time"],
            arr["temp"][0, :],
            {"max": "temp"},
        ),
        "press": (
            ["time"],
            arr["press"],
            {"mean": "press"},
        ),
        "cond": (
            ["time"],
            arr["cond"],
            {"max": "cond"},
        ),
    }

    p = ShearProcessing.from_atomix_netcdf(
        atomix_benchmark_baltic_fpath,
        level=1,
        data_aux=data_aux,
    )

    level1 = p.level1
    level2 = p.level2
    level3 = p.level3
    level4 = p.level4

    assert isinstance(level1, ShearLevel1)
    assert isinstance(level2, ShearLevel2)
    assert isinstance(level3, ShearLevel3)
    assert isinstance(level4, ShearLevel4)

    ds1 = xr.load_dataset(atomix_benchmark_baltic_fpath, group="L1_converted")
    ds2 = xr.load_dataset(atomix_benchmark_baltic_fpath, group="L2_cleaned")
    ds3 = xr.load_dataset(atomix_benchmark_baltic_fpath, group="L3_spectra").rename(
        {
            "N_SHEAR_SENSORS": "nshear",
            "SH_SPEC": "psi_k_sh",
            "KCYC": "k",
            "WAVENUMBER": "wavenumber",
        }
    )  # for consistency with turban level 3
    ds4 = xr.load_dataset(atomix_benchmark_baltic_fpath, group="L4_dissipation")

    data_tree = p.to_xarray()
    ds1_turban, ds2_turban, ds3_turban, ds4_turban = data_tree.values()

    assert "time" in ds3_turban.coords
    assert "waveno" in ds3_turban.coords
    assert "freq" in ds3_turban.coords

    _plot_level4(ds4, ds4_turban)  # TODO


def _plot_level4(ds4, level4):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.plot(ds4.PRES, ds4.EPSI_FINAL, "k", label="benchmark", marker="o")
    ax.plot(level4.press, level4.eps.mean("nshear"), "r", label="turban", marker="o")
    ax.set_xlabel("Pressure (dbar)")
    ax.set_ylabel("Dissipation rate (W/kg)")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig("out/tests/process/shear/baltic-level4-eps.png")
