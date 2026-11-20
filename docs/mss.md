# Turban Mikrostruktur Sonde (MSS) implementation reference

The MSS is a shear microstructure profiles produced by Sea & Sun Technology (SST). 

See here for a documentation of the development history (link to Matthäus article, to be added soon).

## Technical setup of a MSS

A standard MSS is connected by a 4-wire cable that provides power (2-wires) and a uni-directional
2-wire RS422 connection with 600 kbit [^1].
If the device is powered, it will send raw binary data in the HHL-Format using the RS422 interface.
In a typical setup the MSS is connected to a computer with the SST-SDA software, which is capable to collect
raw MSS data together with a GPS device. SST-SDA writes the data in the MRD format, combining
MSS and optional GPS data in binary format but with a verbose header of the measurement setup.
The typical workflow is to create a configuration for a specific MSS and to process a bunch of MRD files.
This will be explained below. For processing the HHL data refer to HHL API reference below. 


The configuration of an MSS is done via the [`MssDeviceConfig`][turban.instruments.mss.config].

[^1]: Check baud rate!

## Loading data for processing in TURBAN

```python
import numpy as np

from turban import ShearProcessing, ShearLevel1, ShearConfig
from turban.instruments.mss.config import MssDeviceConfig
from turban.instruments.mss.mss_mrd import read_mrd, raw_to_level0, level0_to_level1
from turban.utils.filepaths import atomix_benchmark_baltic_mrd_fpath

# For well-curated .MRD files (including e.g. calibration constants),
# we can load the configuration and metadata directly from there
mss_conf = MssDeviceConfig.from_mrd(
    filename=atomix_benchmark_baltic_mrd_fpath,
    shear_sensitivities={
        "SHE1": 3.90e-4,
        "SHE2": 4.05e-4,
    },  # sensors 32 and 33 (MSS038)
    # link the standard ctd names (turban.variables.STANDARD_CTD_TYPE) to the
    # sensor names in the MRD header; guessing them when omitted is deprecated
    sensornames_ctd={"cond": "COND", "temp": "TEMP", "press": "PRESS"},
    offset=0,
)

# load MSS data
with open(atomix_benchmark_baltic_mrd_fpath, "rb") as f:
    data_raw = read_mrd(f)
data_level0 = raw_to_level0(mss_conf, data_raw)

data_level1 = level0_to_level1(mss_conf, data_level0)

# Configure the TURBAN processing pipeline
shear_config = ShearConfig(
    sampfreq=1024.0,
    segment_length=2048,
    segment_overlap=1024,
    chunk_length=4 * 2048,
    chunk_overlap=1024,
    freq_highpass=1,
    spatial_response_wavenum=50.0,
    spike_threshold=8.0,
    max_tries=10,
    spike_replace_before=512,
    spike_replace_after=512,
    spike_include_before=10,
    spike_include_after=20,
    cutoff_freq_lp=0.5,
)
section_number = np.asarray(data_level1["time_count"] * 0, dtype=int)
section_number[2677:152620] = int(1) # manually defines a section of interest
shear_level1 = ShearLevel1(
    time=np.asarray(data_level1["time_count"]),
    senspeed=np.asarray(data_level1["PSPD_REL"]),
    shear=np.asarray(data_level1["SHEAR"]),
    section_number=section_number,
    cfg=shear_config,
)

# add auxiliary variables not strictly necessary for TURBAN
for v in ["PRESS", "TEMP", "COND"]:
    shear_level1.add_aux_data(data_level1[v].values, v.lower())

# set up and run the pipeline
p = ShearProcessing(shear_level1)
# p now contains level1. When a higher level is called,
# the actual computations up until that level start happening, e.g.:
ds_level3 = p.level3.to_xarray()
```

We can now automagically plot the entire pipeline or any parts of it:

```python continuation
from turban import plot
# _ = plot(p.level2)  # would also work
# _ = plot(p.level3.to_xarray())  # would also work

# since p contains all four levels, the following call produces one figure for each
figs = plot(p)
assert len(figs) == 4

for i, (fig, axes) in enumerate(figs):
    fig.savefig(f"out/tests/instruments/mss/SH2_0330_level{i+1}.png")
```

When the .MRD file is not as well curated or for any other reason, we can load from a json file as well.
`MssDeviceConfig` is a pydantic model, so a configuration read with `from_mrd` round-trips
through a python dict or JSON without loss:

```python continuation
import json

# save an existing configuration
with open(f"out/tests/instruments/mss/SH2_0330_config.json", "w") as f:
    f.write(mss_conf.model_dump_json(indent=4))

# load back in again
with open(f"out/tests/instruments/mss/SH2_0330_config.json", "r") as f:
    mss_conf2 = MssDeviceConfig.model_validate(json.load(f))

assert mss_conf == mss_conf2
```

## One-call loading with `mrd_to_level1`

`turban.instruments.mss.api.mrd_to_level1` performs all of the above steps (reading, Level 0, Level 1) in one call and returns a TURBAN Level 1 object, either `ShearLevel1` (`target="shear"`) or `UTempLevel1` (`target="utemp"`):

```python
from turban import ShearConfig, ShearProcessing
from turban.instruments.mss.api import mrd_to_level1
from turban.utils.filepaths import atomix_benchmark_baltic_mrd_fpath

cfg = ShearConfig(
    sampfreq=1024.0,
    segment_length=2048,
    segment_overlap=1024,
    chunk_length=4 * 2048,
    chunk_overlap=1024,
)
level1 = mrd_to_level1(
    atomix_benchmark_baltic_mrd_fpath,
    "shear",
    cfg,
    shear_sensitivities={"SHE1": 3.90e-4, "SHE2": 4.05e-4},
    sensornames_ctd={"cond": "COND", "temp": "TEMP", "press": "PRESS"},
)
assert "temp" in level1.aux_data
```

The device configuration is either read from the MRD header as with `MssDeviceConfig.from_mrd` (then `shear_sensitivities` is required and `sensornames_ctd` should be given), or passed as a `MssDeviceConfig` (fourth argument `mss_cfg`), which takes precedence.

`mrd_to_level1`

- uses the physical time if the MRD file contains a time packet, and otherwise the seconds since the first sample,
- sets `section_number` to 1 for the whole file, i.e. all data are processed. To restrict processing to e.g. the downcast, set `level1.section_number` (0 means discard) before setting up the pipeline, see [Sections, segments, and chunks](manual.md#sections-segments-and-chunks),
- adds the CTD sensors listed in `sensornames_ctd` as auxiliary data under their standard names `cond`, `temp` and `press`. Because `temp` is present, Level 4 computes the molecular viscosity from it (see [Molecular properties of seawater](manual.md#molecular-properties-of-seawater)),
- adds the channels `ACC`, `Chl_A`, `ACCx` and `ACCy` as auxiliary data, if present.

### Missing data records

Records missing from an MRD file (e.g. due to transmission errors) are detected from gaps in the `COUNT` channel. The argument `missing` of `mrd_to_level1` (and `mss_mrd.level0_to_level1`) controls how they are handled:

| `missing` | Effect |
|---|---|
| `None` (default) | Records are kept as they are, so the time axis is irregular at gaps |
| `"interpolate"` | Missing records are inserted and linearly interpolated in the calibrated channels before the derived quantities (salinity, density, sinking speed, shear, microtemperature) are computed |
| `"nan"` | As `"interpolate"`, but the inserted records are then set to NaN in all variables |

With `"interpolate"` and `"nan"`, duplicate records are dropped, and a boolean variable `inserted` marks the inserted records in the Level 1 dataset of `level0_to_level1`. Gaps of 65536 records or more cannot be detected.

### Microtemperature

With `target="utemp"`, `mrd_to_level1` returns a `UTempLevel1` for the [microtemperature pipeline](utemp.md) (experimental). The fast thermistor channel of the MSS is pre-emphasised (`x + g dx/dt`), and the Level 1 conversion removes this pre-emphasis from the `NTC`/`NTCHP` channels with a digital first-order low-pass (`mss_utils.deconvolve_mss_ntchp`). The time constant $g$ of the pre-emphasis is set by `MssDeviceConfig.gain_utemp` (default 1.5 s). As the digital filter differs from the ideal analog one at high frequencies, `mrd_to_level1` sets `UTempLevel1.spectral_correction` accordingly (`turban.process.utemp.util.correction_frequency_response_bilinear`), on the frequency grid given by the `segment_length` and `sampfreq` of the `UTempConfig`.

```python
import json
from turban.instruments.mss.api import mrd_to_level1
from turban.instruments.mss.config import MssDeviceConfig
from turban.process.utemp.api import UTempProcessing
from turban.process.utemp.config import UTempConfig
from turban.utils.filepaths import mss_probeconf_json_fpath, mss_utemp_mrd_fpath

with open(mss_probeconf_json_fpath) as f:
    mss_cfg = MssDeviceConfig.model_validate(json.load(f))

cfg = UTempConfig(
    sampfreq=1024.0,
    segment_length=2048,
    segment_overlap=1024,
    chunk_length=4 * 2048,
    chunk_overlap=1024,
    waveno_limit_upper=500.0,
)
level1 = mrd_to_level1(mss_utemp_mrd_fpath, "utemp", cfg, mss_cfg)
p = UTempProcessing(level1)  # p.level4 contains chi and eps
```

## MSS API-Reference

See the [API reference](api.md) for the following modules:

- `turban.instruments.mss.api`: `mrd_to_level1`
- `turban.instruments.mss.config`: `MssDeviceConfig` and sensor configurations
- `turban.instruments.mss.mss_mrd`: MRD (MSS Raw Data) file format
- `turban.instruments.mss.mss_hhl`: HHL (High-High-Low) raw data format
- `turban.instruments.mss.mss_utils`: shear, sinking speed and deconvolution helpers






