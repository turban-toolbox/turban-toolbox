# Turbulence analysis (TURBAN) toolbox: Infrastructure

This manual assumes that you have read your data into python. For converting common file formats to python data, see the documentation of the individual instruments and platforms (such as [MSS](mss.md) or [MicroRider](urider.md)).

## Exporting and importing data

TURBAN can export to and import from xarray datasets using its `.to_xarray()` and `.from_xarray()` methods available on `Processing` as well as `Level1/2/3/4` objects. 
TURBAN further defines convenience methods for importing ATOMIX files, which have been tested on a few of the available benchmark files. However, these tend to not follow a 100% strict format and so these methods may fail in untested cases.

The following would do a roundtrip in TURBAN:
```python
from turban import ShearLevel3
l3 = ShearLevel3.from_atomix_netcdf("data/process/shear/MSS_Baltic.nc")  # import from benchmark file 
ds = l3.to_xarray()  # TURBAN-compliant xarray dataset
l3_reimport = ShearLevel3.from_xarray(ds)  # equal to l3
```
## Shear processing

### Removal of coherent vibrations

TURBAN has implemented the Goodman method as described in the ATOMIX shear paper. It is not activated by default since vibration channels are not required, but can be achieved in this way:

```python
import xarray as xr
from turban import ShearLevel2, ShearLevel3
from turban.utils.filepaths import atomix_benchmark_faroe_fpath

ds = xr.load_dataset(atomix_benchmark_faroe_fpath, group="L2_cleaned")
# now, add as many channels as desired to detect coherent vibrations
l2 = ShearLevel2.from_atomix_netcdf(atomix_benchmark_faroe_fpath)
l2.add_aux_data(ds.ACC.isel(N_ACC_SENSORS=0).values, "vib1")
l2.add_aux_data(ds.ACC.isel(N_ACC_SENSORS=1).values, "vib2")
l2.add_aux_data(ds.ACC.isel(N_ACC_SENSORS=2).values, "vib3")
l2.cfg.vibration_channels = ("vib1", "vib2", "vib3")
# now, propagation to ShearLevel3 will automatically remove these 
l3 = ShearLevel3.from_level_below(l2)
```

### Molecular viscosity

Molecular viscosity can be set in two ways: Either by using a constant fallback value in the `ShearConfig`:

```{.python notest}
# Option 1
from turban import ShearConfig
cfg = ShearConfig(
    molvisc_fallback=1.6e-6,
    ...
)
```

or by explicitly setting a `molvisc` auxiliary variable on the `ShearLevel3` object from which Level 4 is derived. This can for instance be achieved by setting an auxiliary variable on Level 1 with the appropriate aggregation instructions:

```python
# Option 2
import numpy as np
from turban import ShearLevel1

level1 = ShearLevel1.from_atomix_netcdf("data/process/shear/MSS_Baltic.nc")
molvisc_arr = np.linspace(1e-6, 2e-6, len(level1.time))
# specify aggregated name `molvisc` 
level1.add_aux_data(molvisc_arr, "molvisc", "mean", "molvisc")
# level 3 will now contain a chunk-averaged variable called `molvisv`,
# which level 4 will then use
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
2. Processing pipelines, defined in `turban/process`, e.g. `ShearProcessing` or one of its levels (e.g., `ShearLevel1` through `ShearLevel4`), are configured using `turban.process.generic.config.SegmentConfig`

Both allow/require the user to make various parameter choices.

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
