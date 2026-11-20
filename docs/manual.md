# Turbulence analysis (TURBAN) toolbox: Infrastructure

This manual assumes that you have read your data into python. For converting common file formats to python data, see the documentation of the individual instruments and platforms (such as [MSS](mss.md) or [MicroRider](microrider.md)).

## Exporting and importing data

TURBAN can export to and import from xarray using `.to_xarray()` and `.from_xarray()`, available on both levels (e.g. `ShearLevel1` through `ShearLevel4`) and pipelines (e.g. `ShearProcessing`). They differ in scope:

- **Levels** export to, and parse from, a single `xr.Dataset` holding only that level. `level_below` is neither exported nor reconstructed. Auxiliary data that Level 2/4 take over from Level 1/3 are included in the export, so the dataset is complete by itself.
- **Pipelines** export to, and parse from, an `xr.DataTree` with nodes `level1` through `level4`, i.e. the whole pipeline including the `level_below` hierarchy. Auxiliary data shared between levels are stored only once (in the `level1` and `level3` nodes). `.to_netcdf()` and `.from_netcdf()` do the same with netCDF files.

A level can also be parsed from such a DataTree, or a netCDF file written by `Processing.to_netcdf()`. This reads only that level's node (plus the auxiliary data it takes over from the level below), and `level_below` is again not reconstructed.

TURBAN further defines convenience methods for importing ATOMIX files, which have been tested on a few of the available benchmark files. However, these tend to not follow a 100% strict format and so these methods may fail in untested cases. 
 
The following round-trips a single level:
```python
from turban import ShearLevel3
l3 = ShearLevel3.from_atomix_netcdf("data/process/shear/MSS_Baltic.nc")  # import from benchmark file 
ds = l3.to_xarray()  # TURBAN-compliant xarray dataset
l3_reimport = ShearLevel3.from_xarray(ds)  # equal to l3
assert l3_reimport.level_below is None  # levels below are not part of the dataset
```

and the following round-trips an entire pipeline:
```python
from pathlib import Path
from tempfile import TemporaryDirectory
from turban import ShearProcessing, ShearLevel4

p = ShearProcessing.from_atomix_netcdf("data/process/shear/MSS_Baltic.nc", level=1)
data_tree = p.to_xarray()  # xr.DataTree with nodes level1 ... level4
p_reimport = ShearProcessing.from_xarray(data_tree)  # entire level_below hierarchy
assert p_reimport.level4.level_below is p_reimport.level3

with TemporaryDirectory() as tmpdir:
    fname = str(Path(tmpdir) / "baltic.nc")
    p.to_netcdf(fname)
    p_reimport = ShearProcessing.from_netcdf(fname)  # same as above
    l4 = ShearLevel4.from_netcdf(fname)  # only the level4 node, without level_below
```

### Variable names, vocabularies and metadata

TURBAN uses standard variable names throughout (`eps`, `senspeed`, `psi_k_sh`, ...). Their metadata are kept in *vocabularies*, i.e. dictionaries of the form `{varname: {attribute: value}}`:

| Vocabulary | Module | Content |
|---|---|---|
| `VARIABLES` | `turban.variables` | Variables shared by all processing types (`time`, `freq`, `waveno`, `eps`, `molvisc`, `temp`, ...) |
| `VARIABLES_SHEAR` | `turban.process.shear.variables` | Shear-specific variables (`shear`, `fom`, `quality_metric`, `log_psi_std`, ...) |
| `VARIABLES_UTEMP` | `turban.process.utemp.variables` | Microtemperature-specific variables (`kbatch`, `likelihood_ratio`, ...) |

The attributes follow the CF conventions where possible: `units`, `standard_name` and `long_name`. TURBAN adds `explanation` (a plain-language description), `latex` (a symbol, used e.g. for plot labels) and `atomix_name` (the name of the corresponding variable in ATOMIX netCDF files).

`to_xarray()` attaches these attributes to every exported variable. Each level class has a default list of vocabularies in its class attribute `vocabulary` (e.g. `[VARIABLES, VARIABLES_SHEAR]` for the shear levels). Pass a different list to `to_xarray()` (on a level or a pipeline) to add or override attributes; later vocabularies take precedence:

```python
from turban import ShearLevel3
from turban.variables import VARIABLES
from turban.process.shear.variables import VARIABLES_SHEAR

l3 = ShearLevel3.from_atomix_netcdf("data/process/shear/MSS_Baltic.nc")
assert l3.to_xarray()["waveno"].attrs["units"] == "cpm"

my_vocabulary = {"psi_k_sh": {"long_name": "shear wavenumber spectrum"}}
ds = l3.to_xarray([VARIABLES, VARIABLES_SHEAR, my_vocabulary])
assert ds["psi_k_sh"].attrs["long_name"] == "shear wavenumber spectrum"
```

For instrument data, `turban.variables.STANDARD_CTD_TYPE` defines the standard names of CTD-type auxiliary variables (`cond`, `press`, `temp`) that instrument sensors are mapped to, see e.g. `MssDeviceConfig.sensornames_ctd` in the [MSS reference](mss.md).

## Shear processing

### Configuration

`ShearConfig` (like all processing configurations, a child class of `SegmentConfig`) is a pydantic model. Some of its parameters have derived defaults:

- `freq_highpass=None` (the default) sets the high-pass cutoff frequency to $1/(2T)$, where $T$ = `segment_length / sampfreq` is the duration of an FFT segment. This value is computed once, when the configuration is created.
- `log_psi_std` ($\sigma_{\ln\Psi}$, Eq. 17 in the ATOMIX paper) and `number_signals_vibration_removal` ($N_V$) are read-only properties derived from the segment parameters and `vibration_channels`.

Assigning to an attribute of an existing configuration is validated, i.e. invalid values raise a `pydantic.ValidationError`:

```python
from turban import ShearConfig

cfg = ShearConfig(
    sampfreq=1024.0,
    segment_length=2048,
    segment_overlap=1024,
    chunk_length=4 * 2048,
    chunk_overlap=1024,
)
assert cfg.freq_highpass == 0.25  # 1 / (2 * 2048 / 1024 Hz)

cfg.vibration_channels = "ACC1"  # validated and converted on assignment
assert cfg.vibration_channels == ("ACC1",)
assert cfg.number_signals_vibration_removal == 1
print(f"sigma_lnPsi = {cfg.log_psi_std:.3f}")
```

!!! warning
	Since the default `freq_highpass` is computed when the configuration is created, changing `segment_length` or `sampfreq` afterwards does not update it. Set `freq_highpass` explicitly in that case.

### Removal of coherent vibrations

TURBAN has implemented the Goodman method as described in the ATOMIX shear paper. Going from Level 2 to Level 3 removes the noise coherent with the aux variables named in `cfg.vibration_channels`. Their number $N_V$ enters `ShearConfig.log_psi_std` ($\sigma_{\ln\Psi}$). Every channel in `vibration_channels` must be present in the aux data, otherwise creating Level 3 raises a `KeyError`. A warning is logged if there are too few FFT segments per chunk for the number of vibration channels ($N_f - N_V \le 3$, Sec. 3.3.4 in the ATOMIX paper).

By default, `vibration_channels` is empty, i.e. no vibration removal. Loading from an ATOMIX netCDF file, however, sets the vibration channels as implied by the file itself: the ATOMIX loaders of the levels (`ShearLevelN.from_atomix_netcdf`) set up the vibration signals in the file (`VIB`, else `ACC`) as aux data `VIB1, VIB2, ...` (`ACC1, ...`) and accordingly `vibration_channels`, so that processing from an ATOMIX Level 1 or 2 removes vibrations with these signals. They are taken from `L1_converted`, else `L2_cleaned`, at Level 1, and from `L2_cleaned`, else Level 1, at Level 2. `ShearConfig.from_atomix_netcdf` alone leaves `vibration_channels` empty, as it has no aux data. `ShearLevel3.from_atomix_netcdf` loads the cleaned spectra (`SH_SPEC_CLEAN`) if present.

```python
from turban import ShearLevel2, ShearLevel3
from turban.utils.filepaths import atomix_benchmark_faroe_fpath

l2 = ShearLevel2.from_atomix_netcdf(atomix_benchmark_faroe_fpath)
assert l2.cfg.vibration_channels == ("ACC1", "ACC2", "ACC3")
# any aux variables added with add_aux_data can be used, () switches removal off
l2.cfg.vibration_channels = ("ACC1", "ACC2")
l3 = ShearLevel3.from_level_below(l2)
```

### Molecular properties of seawater

Level 4 needs the kinematic molecular viscosity $\nu$ of seawater (and the chi pipeline also the molecular thermal diffusivity $\kappa$). TURBAN determines them as follows:

1. If the Level 3 object has an auxiliary variable `temp` (in-situ temperature in °C), $\nu$ (and $\kappa$) are computed from its chunk averages.
2. Otherwise, the constant `molvisc_fallback` (default $1.6\cdot10^{-6}$ m² s⁻¹) of the configuration is used (and `moldiff_fallback` for $\kappa$ in `UTempConfig`).

In both cases, Level 4 stores the values used as `molvisc` (and `moldiff`), one per chunk.

```python
import numpy as np
from turban import ShearLevel3, ShearLevel4
from turban.utils.seawater import viscosity_kinematic

l3 = ShearLevel3.from_atomix_netcdf("data/process/shear/MSS_Baltic.nc")
temp = np.linspace(5.0, 10.0, len(l3.time))  # [degC], one value per chunk
l3.add_aux_data(temp, "temp")
# on Level 1, this would be
# level1.add_aux_data(temp_fast, "temp", "mean", "temp")

l4 = ShearLevel4.from_level_below(l3)
assert np.allclose(l4.molvisc, viscosity_kinematic(temp))
```

The underlying functions are in `turban.utils.seawater`:

| Function | Quantity | Notes |
|---|---|---|
| `viscosity_kinematic(temp)` | $\nu$ [m² s⁻¹] | Polynomial fit at salinity 35; this is what the pipelines use |
| `viscosity_kinematic_full(temp, sal)` | $\nu$ [m² s⁻¹] | Sharqawy et al. (2010), at arbitrary salinity; for reference, not used by the pipelines |
| `diffusivity_temp(temp, sal=35.0)` | $\kappa$ [m² s⁻¹] | Thermal conductivity / (density × specific heat); the pipelines use the default salinity |

There is currently no option to pass salinity to the pipelines. To use a constant viscosity instead, do not provide `temp` and set `molvisc_fallback`.

### Figure of merit and its ingredients

ATOMIX defines as a poor figure of merit (FOM), i.e. $\mathrm{FOM} = \mathrm{MAD} / (\sigma_{\ln\Psi} T_M) > 1.4$, where the threshold of 1.4 is empirical. `ShearLevel4` stores the FOM as `fom` (and exports it with `to_xarray()`), as well as the ingredients needed to recompute it (`waveno_cutoff`, `log_diss_mad`, `num_spec_points`), including `log_psi_std` from the config (`ShearConfig.log_psi_std`):

```python
import numpy as np
from turban import ShearLevel3, ShearLevel4
from turban.process.shear.level4 import figure_of_merit

l3 = ShearLevel3.from_atomix_netcdf("data/process/shear/MSS_Baltic.nc")
l4 = ShearLevel4.from_level_below(l3)

fom, log_diss_mad, num_spec_points = figure_of_merit(
    waveno=l3.waveno,
    waveno_cutoff=l4.waveno_cutoff,  # upper wavenumber limit of the eps estimate
    eps=l4.eps,
    molvisc=l4.molvisc,
    psi=l3.psi_k_sh,
    log_psi_std=l3.cfg.log_psi_std,  # sigma_{ln Psi}
)
# T_M in the ATOMIX paper
tm = 0.8 + 1.25 / np.sqrt(num_spec_points)  
assert np.allclose(fom, l4.fom, equal_nan=True)
```


## Sections, segments, and chunks

### Nomenclature

Large parts of TURBAN operate on data in the form of timeseries $x(t)$ which are sampled at regular time steps $\Delta t=1/f_s$, yielding timeseries numbers $x_i$ where $i=1, 2,\dots N\in\mathbb{N}$. Any such timeseries must be split up at different levels to accommodate e.g. spectral analysis:

1. _Sections_ consist of (not necessarily, but usually) contiguous regions of $x_i$ to be analysed together, such as consecutive dives in the same file, or parts of the same cast that was interrupted by a snagged cable. They may be as long or short as necessary (but if they are shorter than the _chunk_ length, they are effectively discarded from analysis). _Sections_ are specified as integer array with a unique identifier for each _section_.
2. _Segments_ consist of a fixed-length piece of data to be analysed by some method, e.g., FFT, and are thus usually a power of $2$. These are specified by `segment_length`.
3. _Chunks_ consist of multiple consecutive _segments_ in order to e.g. enhance the statistical reliability of a dissipation estimate. Inside each chunk

While sections in practice are mutually exclusive, both segments and chunks often overlap with the previous and next. These are specified in the code as follows:

| Context | Type | Variable name(s) in code| Meaning|
| ---| --- | ---| --- |
|Segment|`int` |`segment_length`| Number of samples per segment |
|Chunk|`int` |`segment_overlap`| Overlap (number of samples) of consecutive segments inside chunk |
|Chunk|`int`| `chunk_length`|Number of samples per individual chunk |
|Section|`int` | `chunk_overlap`|Overlap (number of samples) of consecutive chunks |
|Section|`int` array| `section_number`|Unique identifier per section, `0` means discard|

As example, let us consider a timeseries `x` of 13 samples. (We choose few samples in order to be able to write out arrays.) Of these, the first sample and the last two samples are to be discarded. This is achieved by
```python
import numpy as np
x = np.random.rand(13)
section_number = np.zeros_like(x, dtype=int)
section_number[1:12] = 1
```
leaving us with 10 samples. We would like to compute dissipation using half-overlapping FFT _segments_ of 2 samples, taking 3 such FFT _segments_ for each dissipation estimate (_chunk_), and an overlap of half a FFT _segment_ between _chunks_. Then:
```{.python continuation}
segment_length = 2
segment_overlap = 1
chunk_length = 4 # We can fit 3 half-overlapping segments in here
chunk_overlap = 1
```

### Implementation

In order to use numpy's vectorized routines for time series analysis, we use the function `get_chunking_index`. It takes in the parameters `section_number_or_data_len` and arbitrarily many tuples of the form (`length`, `overlap`), and return an array of indices, in the following called `idx`. 

When called with exactly the two tuples (`chunk_length`, `chunk_overlap`) and(`segment_length`, `segment_overlap`), then index `idx` (see sketch), when used to index an axis of length `len(section_number_or_data_len)` or `section_number_or_data_len` of any array, will trigger expansion of the time axis into three axes, that are, in turn: 
1. The slow/reduced/aggregated time axis, i.e. counting chunks.
2. Inside each chunk, counting the number of segments.
3. Inside each segment, counting the samples attributed to each segment.

For instance, given a an array `x` whose last axis has length `samples_len`, we can calculate the FFT over all segments without any loop:
```{.python continuation}
from turban.utils.util import get_chunking_index
idx = get_chunking_index(
    len(x),
    (chunk_length, chunk_overlap),
    (segment_length, segment_overlap),
)
xr = x[..., idx] # reshape time axis. Now the last dimension contains the FFT segments
xr -= xr.mean(axis=-1)[..., np.newaxis]  # subtract mean
xr *= np.hanning(segment_length)[np.newaxis, np.newaxis, :]  # hanning window
Fx = np.fft.rfft(xr) # FFT(x). Now the last dimension contains the frequency axis
```

Similarly, one can easily calculate segment- or chunkwise variance, trends, or other. The same logic is also used for aggregating variables from "fast" to "slow" time.

## Customising TURBAN

### Logging

TURBAN operates with module-level loggers, one for each `.py` file with the respective name. Their loglevel can be set all at once or only pertaining to specific modules. The following sets the log level to `debug` for all code under `turban/instruments/mss/`:

```python
from turban import set_turban_loglevel
set_turban_loglevel("WARNING")  # default
set_turban_loglevel("DEBUG", "turban.instruments.mss")
```

### Configuration objects

TURBAN operates with several kinds of high-level objects, each of which have their own settings:
1. Instruments, child classes of `turban.instruments.generic.api.Instrument`, are configured through instances of (child classes of) `turban.instruments.generic.config.InstrumentConfig`
2. Processing pipelines, defined in `turban/process`, e.g. `ShearProcessing` or one of its levels (e.g., `ShearLevel1` through `ShearLevel4`), are configured using child classes of `turban.process.generic.config.SegmentConfig`, e.g. `ShearConfig` or `UTempConfig`

Both allow/require the user to make various parameter choices. All configuration objects are pydantic models, so they can be serialised to and from JSON (`model_dump_json()`, `model_validate()`). Processing configurations also validate assignments to their attributes, see [Configuration](#configuration).

### Changing algorithms

When a user needs to change an alorithm beyond a simple parameter, there are two options. 
1. Of course, the source code may be modified.
2. Python allows "monkey patching" individual functions (i.e. replacing them with our own implementations), as in:
```python
from turban.process.shear import level3 
from numpy import ndarray, newaxis
from jaxtyping import Float

def my_apply_compensation_highpass(
    x: Float[ndarray, "n_shear time_slow f"],
    freq: Float[ndarray, "f"],
    freq_highpass: float,
) -> Float[ndarray, "f"]:
    correction_factor = np.ones_like(freq) # No highpass frequency compensation
    x *= correction_factor[newaxis, :]
    return correction_factor

# with the following, TURBAN would now use the user-defined function instead
# level3.apply_compensation_highpass = my_apply_compensation_highpass
```
In fact, this method can be used as long as the function signatures of the old and the new functions are the same.
