# Turbulence analysis (TURBAN) toolbox: Infrastructure

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
section_number = np.zeros_like(x, dtype=int)
section_number[1:12] = 1
```
leaving us with 10 samples. We would like to compute dissipation using half-overlapping FFT _segments_ of 2 samples, taking 3 such FFT _segments_ for each dissipation estimate (_chunk_), and an overlap of half a FFT _segment_ between _chunks_. Then:
```python
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
```python
xr = x[..., idx] # reshape time axis. Now the last dimension contains the FFT segments
xr -= xr.mean(axis=-1)[..., newaxis]  # subtract mean
xr *= np.hanning(segment_length)[newaxis, newaxis, :]  # hanning window
Fx = np.fft.rfft(xr) # FFT(x). Now the last dimension contains the frequency axis
```

Similarly, one can easily calculate segment- or chunkwise variance, trends, or other. The same logic is also used for aggregating variables from "fast" to "slow" time.

## Customising TURBAN

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
from numpy import ndarray
from jaxtyping import Float

def my_apply_compensation_highpass(
    x: Float[ndarray, "n_shear time_slow f"],
    freq: Float[ndarray, "f"],
    freq_highpass: float,
) -> Float[ndarray, "f"]:
    correction_factor = np.ones_like(freq) # No highpass frequency compensation
    x *= correction_factor[newaxis, :]
    return correction_factor

level3.apply_compensation_highpass = my_apply_compensation_highpass # TURBAN will now use the user-defined function instead
```
In fact, this method can be used as long as the function signatures of the old and the new functions are the same.

## Code base overview

TURBAN handles a variety of instruments and a variety of methods of analysing them.
- `turban/instruments/`: Instruments provide a way of getting raw data from an instrument up to level 1 (converted to physical units).
- `turban/process/`: Analysis methods provide a way of getting data from level 1 to level 4
- `turban/.../generic/`: Each of `instruments/` and `process/` has one folder `generic` for base and helper classes, in addition to one folder per instrument or process type.
- `turban/.../.../api.py`: Define high-level objects that handle data loading, writing, and processing between levels. These are the objects the end user preferably works with.
- `turban/.../.../config.py`: Define configuration objects that store parameters about processing pipelines or instruments such as sampling rate, high-pass filter cutoff frequencies, and the like.
