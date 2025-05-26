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

def test_load_atomix_netcdf(atomix_nc_filename):
    from turban.process.shear.api import ShearProcessing

    for level in [1, 2, 3]:
        p = ShearProcessing.from_atomix_netcdf(atomix_nc_filename, level=level)
        assert isinstance(p.level4.eps, np.ndarray)


def test_baltic_benchmark(atomix_nc_filename):
    import xarray as xr
    from turban.process.shear.api import (
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
    aux, _ = p.aux.to_xarray()
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

    _plot_level4(ds4, aux, ds4_turban)  # TODO


def _plot_level4(ds4, aux, level4):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.plot(ds4.PRES, ds4.EPSI_FINAL, "k", label="benchmark", marker="o")
    ax.plot(aux.press, level4.eps.mean("nshear"), "r", label="turban", marker="o")
    ax.set_xlabel("Pressure (dbar)")
    ax.set_ylabel("Dissipation rate (W/kg)")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig("out/tests/baltic-level4-eps.png")
