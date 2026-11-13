import pathlib
import re

import numpy as np
import pytest

import turban.instruments.microrider.sensorspeedplugins as plugins
from turban.instruments.microrider import rsIO
from turban.instruments.microrider.rsCommon import MicrostructureRawData
from turban.instruments.microrider.api import (
    MicroriderAPIError,
    MicroriderConfig,
    MicroriderInstrument,
)
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
    return [dict(name="sh1", sens=1e99), dict(name="sh2", sens=1e99)]


@pytest.fixture
def instrument():
    cfg = MicroriderConfig(sampfreq=1.0, sensors={})
    return MicroriderInstrument(cfg=cfg)


@pytest.fixture
def instrumentPostConfigureSensitivity(channel_cfgs):
    cfg = MicroriderConfig(sampfreq=1.0,
                           sensors={},
                           channel_cfgs=channel_cfgs,
                           sensor_speed_plugin='SensorSpeedConstant',
                           sensor_speed_plugin_parameters={'constant_speed':1.0}
                           )
    return MicroriderInstrument(cfg=cfg)

@pytest.fixture
def instrumentPreConfigured():
    cfg = MicroriderConfig(sampfreq=1.0,
                           sensors={},
                           sensor_speed_plugin='SensorSpeedConstant',
                           sensor_speed_plugin_parameters={'constant_speed':1.0})
    return MicroriderInstrument(cfg=cfg)



# Tests that explicitly set the sensor speed plugin
def test_sensor_speed_constant(instrument, data_path, shear_cfg):
    instrument.set_sensor_speed_plugin(plugins.SensorSpeedConstant(constant_speed=1.0))
    level1 = instrument.file_to_shear_level1(str(data_path), cfg=shear_cfg)
    assert level1.shear[0].std() == _SHEAR_STD


def test_sensor_speed_emc(instrument, data_path, shear_cfg):
    instrument.set_sensor_speed_plugin(plugins.SensorSpeedEMC())
    level1 = instrument.file_to_shear_level1(str(data_path), cfg=shear_cfg)
    assert level1.senspeed.shape == level1.time.shape
    assert level1.senspeed.mean() == _EMC_SPEED_MEAN


def test_sensor_speed_lookup_table(instrument, data_path, shear_cfg):
    _t = np.linspace(_T0, _T1)
    _U = np.ones_like(_t)
    plugin = plugins.SensorSpeedLookupTable()
    plugin.from_timeseries(_t, _U)
    instrument.set_sensor_speed_plugin(plugin)
    level1 = instrument.file_to_shear_level1(str(data_path), cfg=shear_cfg)
    assert level1.shear[0].std() == _SHEAR_STD


def test_sensor_speed_data_file(instrument, data_path, shear_cfg, tmp_path):
    _t = np.linspace(_T0, _T1)
    _U = np.ones_like(_t)
    speed_file = tmp_path / "testdata.dat"
    np.savetxt(speed_file, np.array((_t, _U)).T)
    instrument.set_sensor_speed_plugin(plugins.SensorSpeedDataFile(str(speed_file)))
    level1 = instrument.file_to_shear_level1(str(data_path), cfg=shear_cfg)
    assert level1.shear[0].std() == _SHEAR_STD


def test_custom_plugin(instrument, data_path, shear_cfg):
    @plugins.register_plugin
    class SensorSpeedTestCustom(plugins.SensorSpeedABC):
        def __init__(self, scale: float) -> None:
            self._scale = scale

        def get_sensor_speed(self, t):
            return np.ones_like(t) * self._scale

    instrument.set_sensor_speed_plugin(SensorSpeedTestCustom(scale=1.0))
    level1 = instrument.file_to_shear_level1(str(data_path), cfg=shear_cfg)
    assert level1.shear[0].std() == _SHEAR_STD


# Tests by configuration speed plugin
def test_sensor_speed_constant_pre_configured(instrumentPreConfigured, data_path, shear_cfg):
    level1 = instrumentPreConfigured.file_to_shear_level1(str(data_path), cfg=shear_cfg)
    assert level1.shear[0].std() == _SHEAR_STD

    
def test_sensor_sensitivity(instrumentPostConfigureSensitivity, data_path, shear_cfg):
    level1 = instrumentPostConfigureSensitivity.file_to_shear_level1(str(data_path), cfg=shear_cfg)
    assert np.abs(level1.shear[0].mean()) < np.float64(1e-99) # should be very close to 0.
    assert np.abs(level1.shear[1].mean()) < np.float64(1e-99)


def test_requesting_nonexisting_plugin():
    with pytest.raises(plugins.PluginError):
        plugin = plugins.get_registered_plugin_parameter_list("non_exisiting")


# Malformed plugin names must give a PluginError, not whatever importlib raises.
@pytest.mark.parametrize("name", [":Foo", "some_module:", ".foo:Bar", "a:b:c"])
def test_invalid_plugin_name_raises_plugin_error(name):
    with pytest.raises(plugins.PluginError):
        plugins.get_registered_plugin_parameter_list(name)


def test_unimportable_plugin_module_raises_plugin_error():
    with pytest.raises(plugins.PluginError, match="no_such_module_xyz"):
        plugins.get_registered_plugin_parameter_list("no_such_module_xyz:Foo")


# A plugin module that exists but fails on one of its own imports must say which
# import failed.
def test_import_error_inside_plugin_module_is_reported(tmp_path, monkeypatch):
    (tmp_path / "plugin_with_missing_dependency.py").write_text(
        "import no_such_dependency_xyz\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(plugins.PluginError, match="no_such_dependency_xyz"):
        plugins.get_registered_plugin_parameter_list(
            "plugin_with_missing_dependency:Foo"
        )


def test_plugin_factory_unknown_plugin_raises_plugin_error():
    with pytest.raises(plugins.PluginError):
        plugins.plugin_factory("Nope", {})

def test_overwriting_plugin_prints_log(instrumentPreConfigured, caplog):
    instrument = instrumentPreConfigured
    _t = np.linspace(_T0, _T1)
    _U = np.ones_like(_t)
    plugin = plugins.SensorSpeedLookupTable()
    plugin.from_timeseries(_t, _U)
    with caplog.at_level("WARNING"):
        instrument.set_sensor_speed_plugin(plugin)
    record = caplog.records[0]
    assert record.levelname=="WARNING" and record.message.startswith("Overwriting sensor speed plugin:")


# Missing required plugin parameters must be named in the error log.
def test_missing_plugin_parameter_is_logged(caplog):
    cfg = MicroriderConfig(sampfreq=1.0,
                           sensors={},
                           sensor_speed_plugin="SensorSpeedConstant",
                           sensor_speed_plugin_parameters={})
    with caplog.at_level("ERROR"), pytest.raises(MicroriderAPIError):
        MicroriderInstrument(cfg=cfg)
    messages = [r.message for r in caplog.records if r.levelname == "ERROR"]
    assert messages == [
        "Required parameter constant_speed for SensorSpeedConstant is not configured."
    ]


def test_only_missing_plugin_parameters_are_logged(caplog):
    @plugins.register_plugin
    class SensorSpeedTestMissingParameters(plugins.SensorSpeedABC):
        def __init__(self, p_a: float, p_b: float, p_c: float) -> None:
            super().__init__()

        def get_sensor_speed(self, t):
            return np.ones_like(t)
    cfg = MicroriderConfig(sampfreq=1.0,
                           sensors={},
                           sensor_speed_plugin=plugins.plugin_name(SensorSpeedTestMissingParameters),
                           sensor_speed_plugin_parameters={"p_b": 5.0})
    with caplog.at_level("ERROR"), pytest.raises(MicroriderAPIError):
        MicroriderInstrument(cfg=cfg)
    errors = " ".join(r.message for r in caplog.records if r.levelname == "ERROR")
    assert "p_a" in errors and "p_c" in errors
    assert "p_b" not in errors


# Built-in plugins are registered under their bare class name, custom plugins
# under "module:ClassName".
def test_plugin_name():
    assert plugins.plugin_name(plugins.SensorSpeedConstant) == "SensorSpeedConstant"

    @plugins.register_plugin
    class SensorSpeedTestName(plugins.SensorSpeedABC):
        def get_sensor_speed(self, t):
            return np.ones_like(t)

    expected = f"{__name__}:SensorSpeedTestName"
    assert plugins.plugin_name(SensorSpeedTestName) == expected
    assert plugins.get_registered_plugin_parameter_list(expected) == []


# Plugin parameters must be set in the configuration, so defaults are refused.
def test_plugin_parameter_with_default_raises_plugin_error():
    with pytest.raises(plugins.PluginError, match="scale has a default value"):

        @plugins.register_plugin
        class SensorSpeedTestDefault(plugins.SensorSpeedABC):
            def __init__(self, scale: float = 1.0) -> None:
                super().__init__()

            def get_sensor_speed(self, t):
                return np.ones_like(t)

    with pytest.raises(plugins.PluginError):
        plugins.get_registered_plugin_parameter_list(
            f"{__name__}:SensorSpeedTestDefault"
        )


# Only named parameters can be matched to configured parameters, so *args,
# **kwargs and positional-only parameters are refused.
class _SensorSpeedTestStarArgs(plugins.SensorSpeedABC):
    def __init__(self, *args):
        super().__init__()

    def get_sensor_speed(self, t):
        return np.ones_like(t)


class _SensorSpeedTestStarKwargs(plugins.SensorSpeedABC):
    def __init__(self, **kwargs):
        super().__init__()

    def get_sensor_speed(self, t):
        return np.ones_like(t)


class _SensorSpeedTestPositionalOnly(plugins.SensorSpeedABC):
    def __init__(self, scale, /):
        super().__init__()

    def get_sensor_speed(self, t):
        return np.ones_like(t)


@pytest.mark.parametrize(
    "cls, parameter",
    [
        (_SensorSpeedTestStarArgs, "*args"),
        (_SensorSpeedTestStarKwargs, "**kwargs"),
        (_SensorSpeedTestPositionalOnly, "scale"),
    ],
)
def test_plugin_parameter_not_named_raises_plugin_error(cls, parameter):
    with pytest.raises(
        plugins.PluginError, match=f"parameter {re.escape(parameter)} is not allowed"
    ):
        plugins.register_plugin(cls)
    with pytest.raises(plugins.PluginError):
        plugins.get_registered_plugin_parameter_list(plugins.plugin_name(cls))


# Configured plugin parameters must reach the plugin. Use values other than 1.0,
# so that a plugin that ignored the configuration would not pass unnoticed.
def test_configured_constant_speed_is_used():
    cfg = MicroriderConfig(sampfreq=1.0,
                           sensors={},
                           sensor_speed_plugin="SensorSpeedConstant",
                           sensor_speed_plugin_parameters={"constant_speed": 2.5})
    instrument = MicroriderInstrument(cfg=cfg)
    t = np.linspace(_T0, _T1, 5)
    np.testing.assert_array_equal(instrument.sensor_speed_plugin.get_sensor_speed(t),
                                  np.full_like(t, 2.5))


def test_configured_data_file_is_used(tmp_path):
    _t = np.linspace(_T0, _T1)
    speed_file = tmp_path / "speed.dat"
    np.savetxt(speed_file, np.array((_t, np.full_like(_t, 2.0))).T)
    cfg = MicroriderConfig(sampfreq=1.0,
                           sensors={},
                           sensor_speed_plugin="SensorSpeedDataFile",
                           sensor_speed_plugin_parameters={"filename": str(speed_file)})
    instrument = MicroriderInstrument(cfg=cfg)
    np.testing.assert_allclose(instrument.sensor_speed_plugin.get_sensor_speed(_t), 2.0)


# A plugin that is fed new input must not keep using an interpolator built
# from the previous input (e.g. when processing several files in a row).
def test_emc_plugin_follows_new_raw_data(data_path):
    raw = rsIO.read_p_file(str(data_path))
    t = raw.header.timestamp + raw.header.t_slow
    plugin = plugins.SensorSpeedEMC()

    plugin.set_microstructure_raw_data(raw)
    speed_first = plugin.get_sensor_speed(t)

    # Same file, but with U_EM doubled: the speed must double as well.
    raw_doubled = MicrostructureRawData(
        header=raw.header, data={**raw.data, "U_EM": 2 * raw.data["U_EM"]}
    )
    plugin.set_microstructure_raw_data(raw_doubled)
    speed_second = plugin.get_sensor_speed(t)

    np.testing.assert_allclose(speed_second, 2 * speed_first)


def test_lookup_table_follows_new_timeseries():
    t = np.linspace(_T0, _T1)
    plugin = plugins.SensorSpeedLookupTable()

    plugin.from_timeseries(t, np.full_like(t, 1.0))
    np.testing.assert_allclose(plugin.get_sensor_speed(t), 1.0)

    plugin.from_timeseries(t, np.full_like(t, 2.0))
    np.testing.assert_allclose(plugin.get_sensor_speed(t), 2.0)


def test_channel_cfgs_from_dict_are_applied(data_path, shear_cfg):
    cfg = MicroriderConfig(sampfreq=1.0,
                           sensors={},
                           channel_cfgs=[{"name": "sh1", "sens": 1e99},
                                         {"name": "sh2", "sens": 1e99}],
                           sensor_speed_plugin="SensorSpeedConstant",
                           sensor_speed_plugin_parameters={"constant_speed": 1.0})
    level1 = MicroriderInstrument(cfg=cfg).file_to_shear_level1(str(data_path), cfg=shear_cfg)
    assert np.abs(level1.shear[0].mean()) < np.float64(1e-99)  # should be very close to 0.
    assert np.abs(level1.shear[1].mean()) < np.float64(1e-99)


# Without a configured sensor speed plugin, file_to_shear_level1 must raise a clear error.
def test_file_to_shear_level1_without_plugin_raises(instrument, data_path, shear_cfg):
    with pytest.raises(MicroriderAPIError, match="[Ss]ensor speed plugin"):
        instrument.file_to_shear_level1(str(data_path), cfg=shear_cfg)

