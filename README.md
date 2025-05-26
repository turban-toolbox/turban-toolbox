[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Turbulence Analysis Toolbox (Turban-Toolbox)



## Quickstart

Currently, only shear processing is functional. The high-level API can be imported from `turban.shear`.

```
from turban.shear import ShearProcessing

# Process a level 1 dataset all the way to level 4
p = ShearProcessing.from_atomix_netcdf("data/mss/MSS_Baltic.nc", level=1)

# extract some aggregated auxiliary data (`slow` timestamp)
ds_slow, _ = p.aux.to_xarray() # the second argument is data for the `fast` timestamp

# merge with level 4 (dissipation data)
ds = p.level4.to_xarray().merge(ds_slow)

ds.plot.scatter(x="press", y="eps", yscale="log")
```

In general, `tests/` and in particular `tests/shear/test_process.py` contain more examples of how to use the high-level data structures.

## Type checking

TURBAN uses extensive type annotations. Types are currently (for easier development) runtime-checked with `beartype`. This can easily be disabled by commenting out the two lines in `turban/__init__.py`.

## PEP8

TURBAN uses black for the code style. Settings are given in `pyproject.toml` (and should be auto-discovered by black). Code editors like VSCode have extensions that can format files individually, another option is running black on the command line: `$ black .`.