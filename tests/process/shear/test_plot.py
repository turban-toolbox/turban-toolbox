import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from turban.process.shear.api import (
    ShearProcessing,
    ShearLevel1,
    ShearConfig,
    ShearLevel2,
)
from turban.utils.plot.shear import plot_level1, plot_level2, plot_level3, plot_level4

from tests.fixtures import atomix_mss_nc_filename


def test_plot(atomix_mss_nc_filename):

    l1 = ShearLevel1.from_atomix_netcdf("data/mss/MSS_Baltic.nc")
    ds = xr.load_dataset("data/mss/MSS_Baltic.nc", group="L1_converted")
    l1.add_aux_data(ds["TEMP"].values.squeeze(), name="temp", agg_method="mean")
    p = ShearProcessing(l1)
    plot_level1(l1)
    plot_level2(p.level2, p.level1)
    plot_level3(p.level3)
    plot_level4(p.level4)
    plt.show()
