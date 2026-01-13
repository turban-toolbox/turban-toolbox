[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Turbulence Analysis Toolbox (Turban-Toolbox)



## Quickstart

### Using the high-level ShearProcessing pipeline
Currently, only shear processing is functional. The high-level API can be imported from `turban.process.shear.api`.

```python
from turban.process.shear.api import ShearProcessing

# Process a level 1 dataset all the way to level 4
p = ShearProcessing.from_atomix_netcdf("data/mss/MSS_Baltic.nc", level=1)

# extract some aggregated auxiliary data (`slow` timestamp)
ds_slow, _ = p.aux.to_xarray() # the second argument is data for the `fast` timestamp

# merge with level 4 (dissipation data)
ds = p.level4.to_xarray().merge(ds_slow)

ds.plot.scatter(x="press", y="eps", yscale="log")
```

### Manual configuration

If no suitable `ShearProcessing.from_*` method exists, you can configure one manually like this (use `time`, `shear`, etc. from anywhere):

```python
import numpy as np
import xarray as xr
from turban.process.shear.api import ShearProcessing, ShearLevel1, ShearConfig

atomix_nc_filename = "data/mss/MSS_Baltic.nc"

cfg = ShearConfig(
    sampfreq=1024.0,
    segment_length=2048,
    segment_overlap=1024,
    chunk_length=5120,
    chunk_overlap=2560,
    freq_cutoff_antialias=999.0,
    freq_cutoff_corrupt=999.0,
    freq_highpass=0.15,
    spatial_response_wavenum=50.0,
    waveno_cutoff_spatial_corr=999.0,
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

time = ds1.TIME.values.astype("datetime64[s]").astype(
    np.float64
)  # time in seconds since epoch
level1 = ShearLevel1(
    time=time,
    senspeed=ds1.PSPD_REL.values,
    cfg=cfg,
    shear=ds1.SHEAR.values,
    section_number=ds2.SECTION_NUMBER.values.astype(int),
)

p = ShearProcessing(level1)
```


In general, `tests/` and in particular `tests/shear/test_process.py` contain more examples of how to use the high-level data structures.


## Type checking

TURBAN uses extensive type annotations. Types are currently (for easier development) runtime-checked with `beartype`. This can easily be disabled by commenting out the two lines in `turban/__init__.py`.

## PEP8

TURBAN uses black for the code style. Settings are given in `pyproject.toml` (and should be auto-discovered by black). Code editors like VSCode have extensions that can format files individually, another option is running black on the command line: `$ black .`.

## Tests

TURBAN uses `pytest` for unit testing:
```bash
python -m pytest
```
Settings are defined in `pyproject.toml`.

Markdown documents with python snippets can be tested as well, e.g.:
```bash
python -m pytest --markdown-docs README.md
```

To generate a test coverage report:
```bash
python -m pytest --markdown-docs README.md --cov=turban --cov-report html
```
