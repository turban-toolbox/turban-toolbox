## Quickstart

Currently, only shear processing is fully functional. The high-level API can be imported from `turban.process.shear.api`.

### Setup

Example data, also needed for the unit tests, can be downloaded using:
```bash
python tests/filepaths.py --download
```

### Using the high-level ShearProcessing pipeline

```python
from turban import ShearProcessing, plot

# Process a level 1 dataset all the way to level 4
p = ShearProcessing.from_atomix_netcdf("data/process/shear/MSS_Baltic.nc", level=1)

# export level 4 to an xarray.Dataset
ds = p.level4.to_xarray()

# or plot e.g. like this:
plot(p)
```

### Manual configuration

If no suitable `ShearProcessing.from_*` method exists, you can configure one manually like this (use `time`, `shear`, etc. from anywhere):

```python
import numpy as np
import xarray as xr
from turban import ShearProcessing, ShearLevel1, ShearConfig, plot

atomix_nc_filename = "data/process/shear/MSS_Baltic.nc"

cfg = ShearConfig(
    sampfreq=1024.0,
    segment_length=2048,
    segment_overlap=1024,
    chunk_length=5120,
    chunk_overlap=2560,
    freq_cutoff_antialias=None,  # None means that this cutoff is not applied
    freq_cutoff_corrupt=None,  # None means that this cutoff is not applied
    freq_highpass=0.15,
    spatial_response_wavenum=50.0,
    waveno_cutoff_spatial_corr=None,  # None means that this cutoff is not applied
    waveno_spectral_min=None,  # None means that this cutoff is not applied
    spike_threshold=8.0,
    max_tries=10,
    spike_replace_before=512,
    spike_replace_after=512,
    spike_include_before=10,
    spike_include_after=20,
    cutoff_freq_lp=0.5,
)

ds1 = xr.load_dataset(atomix_nc_filename, group="L1_converted")
ds2 = xr.load_dataset(atomix_nc_filename, group="L2_cleaned")

level1 = ShearLevel1(
    time=ds1.TIME.values,
    senspeed=ds1.PSPD_REL.values,
    cfg=cfg,
    shear=ds1.SHEAR.values,
    section_number=ds2.SECTION_NUMBER.values.astype(int),
)

p = ShearProcessing(level1)
plot(p)
```

In general, `tests/` and in particular `tests/shear/test_process.py` contain more examples of how to use the high-level data structures.
