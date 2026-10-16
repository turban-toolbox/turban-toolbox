import pathlib

import numpy as np
import pytest

import turban.instruments.microrider.sensorspeedplugins as senspeed
from turban.instruments.microrider.api import MicroriderConfig, MicroriderProbe
from turban.process.shear.api import ShearConfig

# Start and end times for data_0413.p
_T0 = np.float64(1755598536.000539)
_T1 = np.float64(1755599018.9676669)
_SHEAR_STD = np.float64(0.14836711824834306)
#_SHEAR_STD = np.float64(0.02201280177731781)
_EMC_SPEED_MEAN = np.float64(0.4292161395343412)

@pytest.fixture(scope="session")
def data_path():
    return pathlib.Path(__file__).parents[3] / "data/instruments/microrider/data_0413.p"


@pytest.fixture(scope="session")
def shear_cfg():
    return ShearConfig(
        sampfreq=512.0,
        segment_length=1024,
        segment_overlap=512,
        chunk_length=2048,
        chunk_overlap=1024,
        freq_cutoff_antialias=999.0,
        freq_cutoff_corrupt=999.0,
        freq_highpass=0.15,
        spatial_response_wavenum=50.0,
        waveno_cutoff_spatial_corr=999.0,
        spike_threshold=8.0,
        max_tries=10,
        spike_replace_before=256,
        spike_replace_after=256,
        spike_include_before=10,
        spike_include_after=20,
        cutoff_freq_lp=0.5,
    )


@pytest.fixture(scope="session")
def channel_cfgs():
    # partial user overrides: only the set fields are merged into the .P config
    sh1 = {"name": "sh1", "sens": 1e99}
    sh2 = {"name": "sh2", "sens": 1e99}
    return [sh1, sh2]


@pytest.fixture
def probe():
    cfg = MicroriderConfig(sampfreq=1.0, sensors={})
    return MicroriderProbe(cfg=cfg)


@pytest.fixture
def probe_custom_sensitivity(channel_cfgs):
    cfg = MicroriderConfig(sampfreq=1.0, sensors={}, channel_cfgs=channel_cfgs)
    return MicroriderProbe(cfg=cfg)


# Unit tests of the sensor speed functions
def test_senspeed_constant():
    t = np.linspace(0.0, 1.0, 11)
    np.testing.assert_array_equal(senspeed.senspeed_constant(t, 0.6), 0.6)


def test_senspeed_constant_empty():
    assert senspeed.senspeed_constant(np.array([]), 0.6).shape == (0,)


def test_senspeed_from_lookup_table():
    t_table = np.array([0.0, 1.0, 2.0])
    U_table = np.array([0.0, 1.0, 0.0])
    U = senspeed.senspeed_from_lookup_table(np.array([0.5, 1.0, 1.5]), t_table, U_table)
    np.testing.assert_allclose(U, [0.5, 1.0, 0.5])


def test_senspeed_from_lookup_table_mismatched_lengths():
    with pytest.raises(Exception):
        senspeed.senspeed_from_lookup_table(
            np.array([0.5]), np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])
        )


# Tests through MicroriderProbe.to_shear_level1
def test_sensor_speed_constant(probe, data_path, shear_cfg):
    level1 = probe.to_shear_level1(str(data_path), cfg=shear_cfg, senspeed=1.0)
    assert level1.shear[0].std() == _SHEAR_STD


def test_sensor_speed_emc(probe, data_path, shear_cfg):
    level1 = probe.to_shear_level1(str(data_path), cfg=shear_cfg, senspeed="em")
    assert level1.senspeed.shape == level1.time.shape
    assert level1.senspeed.mean() == _EMC_SPEED_MEAN


def test_sensor_speed_lookup_table(probe, data_path, shear_cfg):
    _t = np.linspace(_T0, _T1)
    _U = np.ones_like(_t)
    level1 = probe.to_shear_level1(str(data_path), cfg=shear_cfg, senspeed=(_t, _U))
    assert level1.shear[0].std() == _SHEAR_STD


def test_sensor_speed_invalid_string(probe, data_path, shear_cfg):
    with pytest.raises(Exception):
        probe.to_shear_level1(str(data_path), cfg=shear_cfg, senspeed="foo")


def test_sensor_sensitivity(probe_custom_sensitivity, data_path, shear_cfg):
    level1 = probe_custom_sensitivity.to_shear_level1(
        str(data_path), cfg=shear_cfg, senspeed=1.0
    )
    assert np.abs(level1.shear[0].mean()) < np.float64(1e-99) # should be very close to 0.
    assert np.abs(level1.shear[1].mean()) < np.float64(1e-99)
