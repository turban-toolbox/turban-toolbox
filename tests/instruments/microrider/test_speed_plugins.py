import pathlib

import numpy as np
import pytest

import turban.instruments.microrider.sensorspeedplugins as plugins
from turban.instruments.microrider.api import MicroriderConfig, MicroriderProbe
from turban.instruments.microrider.rsCommon import channel_config_factory
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
    sh1 = channel_config_factory("sh1")
    sh1.update("sens", 1e99)
    sh2 = channel_config_factory("sh2")
    sh2.update("sens", 1e99)
    return [sh1, sh2]


@pytest.fixture
def probe():
    cfg = MicroriderConfig(sampfreq=1.0, sensors={})
    return MicroriderProbe(cfg=cfg)


@pytest.fixture
def probePostConfigureSensitivity(channel_cfgs):
    cfg = MicroriderConfig(sampfreq=1.0,
                           sensors={},
                           channel_cfgs=channel_cfgs,
                           sensor_speed_plugin='SensorSpeedConstant',
                           sensor_speed_plugin_parameters={'constant_speed':1.0}
                           )
    return MicroriderProbe(cfg=cfg)

@pytest.fixture
def probePreConfigured():
    cfg = MicroriderConfig(sampfreq=1.0,
                           sensors={},
                           sensor_speed_plugin='SensorSpeedConstant',
                           sensor_speed_plugin_parameters={'constant_speed':1.0})
    return MicroriderProbe(cfg=cfg)



# Tests that explicitly set the sensor speed plugin
def test_sensor_speed_constant(probe, data_path, shear_cfg):
    probe.set_sensor_speed_plugin(plugins.SensorSpeedConstant(constant_speed=1.0))
    level1 = probe.to_shear_level1(str(data_path), cfg=shear_cfg)
    assert level1.shear[0].std() == _SHEAR_STD


def test_sensor_speed_emc(probe, data_path, shear_cfg):
    probe.set_sensor_speed_plugin(plugins.SensorSpeedEMC())
    level1 = probe.to_shear_level1(str(data_path), cfg=shear_cfg)
    assert level1.senspeed.shape == level1.time.shape
    assert level1.senspeed.mean() == _EMC_SPEED_MEAN


def test_sensor_speed_lookup_table(probe, data_path, shear_cfg):
    _t = np.linspace(_T0, _T1)
    _U = np.ones_like(_t)
    plugin = plugins.SensorSpeedLookupTable()
    plugin.from_timeseries(_t, _U)
    probe.set_sensor_speed_plugin(plugin)
    level1 = probe.to_shear_level1(str(data_path), cfg=shear_cfg)
    assert level1.shear[0].std() == _SHEAR_STD


def test_sensor_speed_data_file(probe, data_path, shear_cfg, tmp_path):
    _t = np.linspace(_T0, _T1)
    _U = np.ones_like(_t)
    speed_file = tmp_path / "testdata.dat"
    np.savetxt(speed_file, np.array((_t, _U)).T)
    probe.set_sensor_speed_plugin(plugins.SensorSpeedDataFile(str(speed_file)))
    level1 = probe.to_shear_level1(str(data_path), cfg=shear_cfg)
    assert level1.shear[0].std() == _SHEAR_STD


def test_custom_plugin(probe, data_path, shear_cfg):
    @plugins.register_plugin([("scale", float, 1.0)])
    class SensorSpeedTestCustom(plugins.SensorSpeedABC):
        def __init__(self, scale: float) -> None:
            self._scale = scale

        def interpolation_factory(self):
            raise NotImplementedError

        def get_sensor_speed(self, t):
            return np.ones_like(t) * self._scale

    probe.set_sensor_speed_plugin(SensorSpeedTestCustom(scale=1.0))
    level1 = probe.to_shear_level1(str(data_path), cfg=shear_cfg)
    assert level1.shear[0].std() == _SHEAR_STD


# Tests by configuration speed plugin
def test_sensor_speed_constant_pre_configured(probePreConfigured, data_path, shear_cfg):
    level1 = probePreConfigured.to_shear_level1(str(data_path), cfg=shear_cfg)
    assert level1.shear[0].std() == _SHEAR_STD

    
def test_sensor_sensitivity(probePostConfigureSensitivity, data_path, shear_cfg):
    level1 = probePostConfigureSensitivity.to_shear_level1(str(data_path), cfg=shear_cfg)
    assert np.abs(level1.shear[0].mean()) < np.float64(1e-99) # should be very close to 0.
    assert np.abs(level1.shear[1].mean()) < np.float64(1e-99)


def test_requesting_nonexisting_plugin():
    with pytest.raises(plugins.PluginError):
        plugin = plugins.get_registered_plugin_parameter_list("non_exisiting")

def test_overwriting_plugin_prints_log(probePreConfigured, caplog):
    probe = probePreConfigured
    _t = np.linspace(_T0, _T1)
    _U = np.ones_like(_t)
    plugin = plugins.SensorSpeedLookupTable()
    plugin.from_timeseries(_t, _U)
    with caplog.at_level("WARNING"):
        probe.set_sensor_speed_plugin(plugin)
    record = caplog.records[0]
    assert record.levelname=="WARNING" and record.message.startswith("Overwriting sensor speed plugin:")


