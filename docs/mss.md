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
from tests.filepaths import atomix_benchmark_baltic_mrd_fpath

# For well-curated .MRD files (including e.g. calibration constants),
# we can load the configuration and metadata directly from there
mss_conf = MssDeviceConfig.from_mrd(
    filename=atomix_benchmark_baltic_mrd_fpath,
    shear_sensitivities={
        "SHE1": 3.90e-4,
        "SHE2": 4.05e-4,
    },  # sensors 32 and 33 (MSS038)
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
    freq_highpass=0.15,
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

When the .MRD file is not as well curated or for any other reason, we can load from a json file as well:

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

## MSS API-Reference

[MSS-Config](instruments/mss/config.md)

[MRD (MSS Raw Data)](instruments/mss/mss_mrd.md)

[HHL (High-High-Low)](instruments/mss/mss_hhl.md)






