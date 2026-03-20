[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Turbulence Analysis Toolbox (Turban-Toolbox)

Welcome to TURBAN. You can find documentation on [readthedocs](https://turban-toolbox.readthedocs.io/en/latest/quickstart/).

Currently, only shear processing is fully functional. It has nascent support for temperature microstructure, and more is on the way (e.g. high-frequency ADCP).

### Installation

To install dependencies with pip for the end user, including packages only needed for the MSS:
```bash
python -m pip install -e . --group mss
```
or, including for the microrider:
```bash
python -m pip install -e . --group mss --group microrider
```

or, for developers, something like:
```bash
python -m pip install -e . --group dev --group mss
```

This can be done in any python environment.