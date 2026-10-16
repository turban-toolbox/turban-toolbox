# MicroriderProbe

`MicroriderProbe` reads a RSI MicroRider binary `.p` file and converts it into a
[`ShearLevel1`][turban.process.shear.api.ShearLevel1] dataclass, ready for the turban
shear processing pipeline. Because sensor speed — the speed of the instrument relative
to the water — depends on how the instrument is deployed, it is passed explicitly to
`to_shear_level1` via the `senspeed` argument.

## Quickstart

```python notest
from turban.instruments.microrider.api import MicroriderProbe, MicroriderConfig
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
microrider_config = MicroriderConfig(sampfreq=512.0, sensors={})

probe = MicroriderProbe(cfg=microrider_config)

# Convert the .p file to ShearLevel1, with a constant sensor speed of 0.6 m/s
level1 = probe.to_shear_level1("path/to/data.p", cfg=cfg, senspeed=0.6)
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

## Sensor speed

Sensor speed is deployment-dependent. A free-falling probe sinks under
gravity; a glider-mounted probe moves along the glider flight
path. The `senspeed` argument of `to_shear_level1` selects the speed source:

| `senspeed` | Description |
|---|---|
| `float` | Constant speed in m/s. Useful for bench tests or platforms with well-known, stable speed. |
| `"em"` | Speed from the MicroRider's electromagnetic current meter channel (`U_EM`), linearly interpolated from the slow onto the fast time grid. |
| `(time, speed)` | Lookup table of numpy arrays (time in s since epoch, speed in m/s), e.g. from a navigation system or DVL, linearly interpolated onto the fast time grid. |

```python notest
level1 = probe.to_shear_level1("path/to/data.p", cfg=cfg, senspeed=0.6)
level1 = probe.to_shear_level1("path/to/data.p", cfg=cfg, senspeed="em")
level1 = probe.to_shear_level1("path/to/data.p", cfg=cfg, senspeed=(time_array, speed_array))
```

A lookup table stored in a two-column text file (time, speed) can be loaded with
`numpy.loadtxt`:

```python notest
import numpy as np

time_array, speed_array = np.loadtxt("path/to/speed.txt").T
```

The underlying functions `senspeed_constant`, `senspeed_from_em` and
`senspeed_from_lookup_table` live in `turban.instruments.microrider.sensorspeedplugins`
and can also be called directly.

---

## Channel configuration

By default `MicroriderProbe` reads sensor parameters (e.g. shear probe sensitivity `sens`)
from the setup string embedded in the `.p` file. You can override individual parameters by
passing a list of dictionaries in `MicroriderConfig.channel_cfgs`, each with the channel
`name` and only the fields you want to override:

```python 
from turban.instruments.microrider.api import MicroriderProbe, MicroriderConfig

microrider_config = MicroriderConfig(
    sampfreq=512.0,
    sensors={},
    channel_cfgs=[
        {"name": "sh1", "sens": 0.0700},  # V / (rad s⁻¹)
        {"name": "sh2", "sens": 0.0970},
    ],
)
probe = MicroriderProbe(cfg=microrider_config)
```

All other parameters continue to be read from the setup string embedded in the `.p`
file. When the `.p` file is read, each channel gets a config class matching its `type`
in the setup string (e.g. `ShearChannel`, `ThermChannel`, `PolyChannel` in
`turban.instruments.microrider.rsConversions`), which also does the conversion into
physical units. The overrides are validated against that class: a field name the
channel type does not have, an invalid value, or a channel name that is not in the
file raises an error. Channels of unknown type, or with an incomplete calibration in
the setup string, are left in counts.
