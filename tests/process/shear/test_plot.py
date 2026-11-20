import numpy as np
import matplotlib.pyplot as plt
import pytest
import xarray as xr

from turban.process.shear.api import (
    ShearProcessing,
    ShearLevel1,
    ShearLevel3,
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
    shplot.plot_level4(p.level4)
    shplot.plot(p)
    figs = shplot.plot(p.level3, p.level4, subset=[("press_mean", 20.0, 30.0)])


@pytest.mark.parametrize("with_level4", [False, True])
@pytest.mark.parametrize("average", [False, True])
def test_plot_level3(average, with_level4):
    """plot_level3 must work with time-averaged spectra."""
    p = ShearProcessing(ShearLevel3.from_atomix_netcdf(atomix_benchmark_faroe_fpath))
    data = (p.level3, p.level4) if with_level4 else (p.level3,)
    fig, axs = shplot.plot_level3(*data, average=average)
    # one spectrum line per time step (or a single one when averaged)
    n_lines = 1 if average else len(p.level3.senspeed)
    assert len(axs[0].lines) >= n_lines
    plt.close(fig)
