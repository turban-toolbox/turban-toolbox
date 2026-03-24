import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from turban.process.shear.api import (
    ShearProcessing,
    ShearLevel1,
    ShearConfig,
    ShearLevel2,
)
from turban.utils.plot import shear as shplot

from turban.utils.filepaths import atomix_benchmark_faroe_fpath


def test_plot():

    l1 = ShearLevel1.from_atomix_netcdf(atomix_benchmark_faroe_fpath)
    ds = xr.load_dataset(atomix_benchmark_faroe_fpath, group="L1_converted")
    l1.add_aux_data(ds["PRES"].values.squeeze(), name="press", agg_method="mean")
    p = ShearProcessing(l1)
    shplot.plot_level1(l1)
    shplot.plot_level2(p.level2, p.level1)
    shplot.plot_level3(p.level3)
    shplot.plot_level4(p.level4)
    shplot.plot(p)
    figs = shplot.plot(p.level3, p.level4, subset=[("press_mean", 20.0, 30.0)])
