from itertools import chain
import os
import logging

import numpy as np
import pytest
from scipy.io import loadmat

from turban.instruments.microrider import rsIO
from pydantic import ValidationError

from turban.instruments.microrider.rsConversions import (
    make_channel_config,
    MicroRiderChannelConfig,
    PassThroughChannel,
    PolyChannel,
    PreEmphasisChannel,
    ShearChannel,
)
from turban.utils.filepaths import microrider_data_directory

logger = logging.getLogger()


datadir = microrider_data_directory

ATOL = 2e-6
RTOL = 1e-5


def extract_data(pParameter, mParameter, pdata, mdata):
    vp = pdata.channel_matrix.channels[pParameter].data
    vm = mdata[mParameter]
    return vp, vm


parameter_table = [
    ("Ax", "Ax"),
    ("Ay", "Ay"),
    ("Gnd", "Gnd"),
    ("Incl_T", "Incl_T"),
    ("Incl_X", "Incl_X"),
    ("Incl_Y", "Incl_Y"),
    ("T1_hires", "T1_fast"),
    ("T2_hires", "T2_fast"),
    ("V_Bat", "V_Bat"),
    ("EMC_Cur", "EMC_Cur"),
    ("T1", "T1_slow"),
    ("T2", "T2_slow"),
    ("PV", "PV"),
    ("U_EM", "U_EM"),
    ("sh1", "sh1"),
    ("sh2", "sh2"),
]
parameter_table_persistor_specific = [("Gnd_2", "Gnd_2"), ("ch255", "ch255")]


def load_data(filename, matfilename):
    fn = os.path.join(datadir, filename)
    pdata = rsIO.read_p_file(fn)
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
    return load_data(filename, matfilename)


@pytest.fixture(scope="session")
def data_0413():
    filename = "data_0413.p"
    matfilename = "data_0413.mat"
    return load_data(filename, matfilename)


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


_SHEAR_SECTION = dict(
    id=8, name="sh1", type="shear", adc_fs=4.096, adc_bits=16, diff_gain=0.94, sens=0.07
)


def test_make_channel_config_selects_class_by_type():
    cfg = make_channel_config(_SHEAR_SECTION)
    assert isinstance(cfg, ShearChannel)
    v = np.array([100, -200, 300], dtype=np.int16)
    expected = (4.096 / 2**16 * v) / (0.94 * 0.07 * 2 * np.sqrt(2))
    assert np.allclose(cfg.convert(v), expected)


def test_make_channel_config_collects_poly_coefs():
    cfg = make_channel_config(dict(name="P", type="poly", coef0=-1.0, coef1=0.01))
    assert isinstance(cfg, PolyChannel)
    assert cfg.coefs == [-1.0, 0.01]


def test_make_channel_config_unknown_type_passes_through():
    cfg = make_channel_config(dict(name="O2", type="o2_43f"))
    assert type(cfg) is PassThroughChannel
    assert cfg.convert(np.array([1, 2], dtype=np.int16)).tolist() == [1.0, 2.0]


def test_make_channel_config_incomplete_calibration_passes_through(caplog):
    section = {k: v for k, v in _SHEAR_SECTION.items() if k != "sens"}
    with caplog.at_level("WARNING"):
        cfg = make_channel_config(section)
    assert type(cfg) is PassThroughChannel
    assert "sh1" in caplog.text


@pytest.mark.parametrize("name,type_", [("T1_dT1", "therm"), ("P_dP", "poly")])
def test_make_channel_config_preemphasis_channel(name, type_):
    cfg = make_channel_config(dict(name=name, type=type_, diff_gain=1.03))
    assert type(cfg) is PreEmphasisChannel
    assert cfg.diff_gain == 1.03


def test_make_channel_config_preemphasis_channel_requires_diff_gain():
    with pytest.raises(ValidationError, match="diff_gain"):
        make_channel_config(dict(name="T1_dT1", type="therm"))



def test_numeric_sn_is_accepted():
    cfg = make_channel_config({**_SHEAR_SECTION, "SN": 487})
    assert isinstance(cfg, ShearChannel)
    assert cfg.sn == "487"


def test_numbers_rejected_for_other_str_fields():
    with pytest.raises(ValidationError, match="cal_date"):
        ShearChannel(**_SHEAR_SECTION, cal_date=2017.1)


def test_make_channel_config_poly_without_coefs_passes_through():
    cfg = make_channel_config(dict(name="PV", type="poly"))
    assert type(cfg) is PassThroughChannel
