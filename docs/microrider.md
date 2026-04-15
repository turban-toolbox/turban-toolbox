# MicroriderSonde

`MicroriderSonde` reads a RSI MicroRider binary `.p` file and converts it into a
[`ShearLevel1`][turban.process.shear.api.ShearLevel1] dataclass, ready for the turban
shear processing pipeline. Because sensor speed — the speed of the instrument relative
to the water — depends on how the instrument is deployed, it is provided through a
swappable **sensor speed plugin**.

## Quickstart

```python
import turban.instruments.microrider.sensorspeedplugins as plugins
from turban.instruments.microrider.api import MicroriderSonde
from turban.instruments.generic.config import InstrumentConfig
from turban.process.shear.api import ShearConfig

# Processing configuration
cfg = ShearConfig(
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

# Instrument configuration
microrider_config = InstrumentConfig(sampfreq=512.0, sensors={})

# Create the sonde and attach a sensor speed plugin
sonde = MicroriderSonde(cfg=microrider_config)
sonde.set_sensor_speed_plugin(plugins.SensorSpeedConstant(constant_speed=0.6))

# Convert the .p file to ShearLevel1
level1 = sonde.to_shear_level1("path/to/data.p", cfg=cfg)
```

`level1` is a `ShearLevel1` instance containing:

- `time` — fast-rate time vector (seconds since file epoch)
- `senspeed` — sensor speed in m/s, one value per time step
- `shear` — array of shape `(2, nsamples)`, one row per shear probe
- `section_number` — all ones (single continuous cast)

!!! note "Shear normalisation"
    The raw shear probe voltage is divided by the square of sensor speed
    (`sh.data / senspeed**2`) to obtain velocity shear in s⁻¹. Accurate sensor
    speed is therefore important for the quality of all downstream dissipation estimates.

---

## Sensor speed plugins

Sensor speed is deployment-dependent. A free-falling dropsonde sinks under gravity; a
glider-mounted sonde moves along the glider flight path; a sonde on an AUV may be towed
at a controlled speed. The plugin system lets you provide the appropriate speed source
without changing any processing code.

All plugins implement `SensorSpeedABC` and must define `get_sensor_speed(t)`, which
accepts a time vector and returns a speed value for every time step.

### `SensorSpeedConstant`

Returns the same speed for all time steps. Useful for bench tests, simulations, or
deployments where the platform speed is well-known and stable.

```python
plugin = plugins.SensorSpeedConstant(constant_speed=0.6)  # 0.6 m/s
sonde.set_sensor_speed_plugin(plugin)
```

| Parameter | Type | Description |
|---|---|---|
| `constant_speed` | `float` | Sensor speed in m/s |

### `SensorSpeedEMC`

Derives sensor speed from the electromagnetic current meter channel (`U_EM`) recorded by
the MicroRider itself. The slow-rate `U_EM` data are linearly interpolated onto the fast
time grid. No constructor arguments are required; the MicroRider data are supplied
automatically when `to_shear_level1` is called.

```python
plugin = plugins.SensorSpeedEMC()
sonde.set_sensor_speed_plugin(plugin)
```

### `SensorSpeedLookupTable`

Accepts a user-supplied time and speed array. Useful when sensor speed has been computed
externally (e.g. from a navigation system or DVL) and is available as numpy arrays.

```python
import numpy as np

plugin = plugins.SensorSpeedLookupTable()
plugin.from_timeseries(t=time_array, U=speed_array)
sonde.set_sensor_speed_plugin(plugin)
```

| Method | Arguments | Description |
|---|---|---|
| `from_timeseries(t, U)` | `t`: time in s, `U`: speed in m/s | Populate the lookup table |

Speed is linearly interpolated onto the fast time grid when `get_sensor_speed` is called.

### `SensorSpeedDataFile`

Reads time and speed from a plain text file with two columns (time in s, speed in m/s),
one row per sample. Internally uses `SensorSpeedLookupTable`.

```python
plugin = plugins.SensorSpeedDataFile(filename="path/to/speed.txt")
sonde.set_sensor_speed_plugin(plugin)
```

The file format expected by `numpy.loadtxt`:

```
# time_s   speed_ms
0.000       0.612
0.002       0.614
0.004       0.611
...
```

| Parameter | Type | Description |
|---|---|---|
| `filename` | `str` | Path to the two-column text file |

---

## Writing a custom plugin

Subclass `SensorSpeedABC` and implement `get_sensor_speed` and `interpolation_factory`:

```python
import numpy as np
from jaxtyping import Float
from typing import Callable
from turban.instruments.microrider.sensorspeedplugins import SensorSpeedABC, register_plugin


@register_plugin([("scale", float, 1.0)])
class SensorSpeedMyCustom(SensorSpeedABC):

    def __init__(self, scale: float) -> None:
        self._scale = scale

    def interpolation_factory(self) -> Callable:
        # build and return an interpolation function, or raise NotImplementedError
        raise NotImplementedError

    def get_sensor_speed(
        self, t: Float[np.ndarray, "time"]
    ) -> Float[np.ndarray, "time"]:
        # return speed at every time step in t
        return np.ones_like(t) * self._scale
```

Decorating with `@register_plugin` makes the class available through the config-driven
factory (see below) and lists it in the registry for introspection.

---

## Config-driven plugin selection

`MicroriderSonde` also supports selecting a plugin through the instrument configuration
object, which is convenient when configuration is loaded from a file rather than
constructed in code. The relevant fields on the config are `sensor_speed_plugin` (the
class name as a string) and `sensor_speed_plugin_parameters` (a dictionary of constructor
arguments). When these are set, `MicroriderSonde` instantiates the plugin automatically
at construction time and no explicit call to `set_sensor_speed_plugin` is needed.

Calling `set_sensor_speed_plugin` after construction always works and overrides any
plugin set from the configuration, with a warning logged.
