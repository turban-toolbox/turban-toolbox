from itertools import chain
import os
import logging

import numpy as np
import pytest
from scipy.io import loadmat

from turban.instruments.microrider import rsIO, channels
from turban.utils.filepaths import microrider_data_directory

logger = logging.getLogger()


datadir = microrider_data_directory

# Tolerances close to machine precision.
ATOL = 1e-12
RTOL = 0


def extract_data(pParameter, mParameter, pdata, mdata):
    vp = pdata[pParameter]
    vm = mdata[mParameter]
    return vp, vm


parameter_table = [
    ("Ax", "Ax"),
    ("Ay", "Ay"),
    ("Gnd", "Gnd"),
    ("Incl_T", "Incl_T"),
    ("Incl_X", "Incl_X"),
    ("Incl_Y", "Incl_Y"),
    ("T1", "T1_fast"),
    ("T2", "T2_fast"),
    ("V_Bat", "V_Bat"),
    ("EMC_Cur", "EMC_Cur"),
    ("PV", "PV"),
    ("U_EM", "U_EM"),
    ("sh1", "sh1"),
    ("sh2", "sh2"),
]


parameter_table_persistor_specific = [("Gnd_2", "Gnd_2"), ("ch255", "ch255")]


def load_data(filename, matfilename, a, b):
    fn = os.path.join(datadir, filename)
    # We need to override the conversion parameters for U_EM, because
    # matlab drops some nth decimal. See also test_rsIO.
    u_em_config = channels.ChannelConfigU_EM(name="U_EM", a=a, b=b)
    pdata = rsIO.read_p_file(fn, channel_configs=[u_em_config])
    mdata = loadmat(
        os.path.join(datadir, matfilename),
        squeeze_me=True,
        mat_dtype=False,
        chars_as_strings=True,
        simplify_cells=True,
    )
    return pdata, mdata


@pytest.fixture(scope="session")
def data_058():
    filename = "DAT_058.P"
    matfilename = "DAT_058.mat"
    a = -24.4605
    b = 151.341
    return load_data(filename, matfilename, a, b)


@pytest.fixture(scope="session")
def data_0413():
    filename = "data_0413.p"
    matfilename = "data_0413.mat"
    a = -25.0675
    b = 0.0114359
    return load_data(filename, matfilename, a, b)


@pytest.mark.parametrize("params", parameter_table)
def test_parameters_persistor(data_058, params):
    vp, vm = extract_data(*params, *data_058)
    d = np.isclose(vp, vm, rtol=RTOL, atol=ATOL)
    assert np.all(d)


@pytest.mark.parametrize("params", parameter_table_persistor_specific)
def test_parameters_persistor_specific(data_058, params):
    vp, vm = extract_data(*params, *data_058)
    d = np.isclose(vp, vm, rtol=RTOL, atol=ATOL)
    assert np.all(d)


@pytest.mark.parametrize("params", parameter_table)
def test_parameters(data_0413, params):
    vp, vm = extract_data(*params, *data_0413)
    d = np.isclose(vp, vm, rtol=RTOL, atol=ATOL)
    assert np.all(d)
