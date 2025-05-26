# Turbulence analysis (TURBAN) toolbox: Infrastructure

## Sections, segments, and chunks

### Nomenclature

Large parts of TURBAN operate on data in the form of timeseries $x(t)$ which are sampled at regular time steps $\Delta t=1/f_s$, yielding timeseries numbers $x_i$ where $i=1, 2,\dots N\in\mathbb{N}$. Any such timeseries must be split up at different levels to accommodate e.g. spectral analysis:

1. _Sections_ consist of (not necessarily, but usually) contiguous regions of $x_i$ to be analysed together, such as consecutive dives in the same file, or parts of the same cast that was interrupted by a snagged cable. They may be as long or short as necessary (but if they are shorter than the _chunk_ length, they are effectively discarded from analysis). _Sections_ are specified as integer array with a unique identifier for each _section_.
2. _Segments_ consist of a fixed-length piece of data to be analysed by some method, e.g., FFT, and are thus usually a power of $2$. These are specified by `segment_len`.
3. _Chunks_ consist of multiple consecutive _segments_ in order to e.g. enhance the statistical reliability of a dissipation estimate. Inside each chunk

While sections in practice are mutually exclusive, both segments and chunks often overlap with the previous and next. These are specified in the code as follows:

| Context | Type | Variable name(s) in code| Meaning|
| ---| --- | ---| --- |
|Segment|`int` |`segment_len`| Number of samples per segment |
|Chunk|`int` |`segment_overlap`| Overlap (number of samples) of consecutive segments inside chunk |
|Chunk|`int`| `chunk_len`|Number of samples per individual chunk |
|Section|`int` | `chunk_overlap`|Overlap (number of samples) of consecutive chunks |
|Section|`int` array| `section_marker`|Unique identifier per section, `0` means discard|

As example, let us consider a timeseries `x` of 13 samples. (We choose few samples in order to be able to write out arrays.) Of these, the first sample and the last two samples are to be discarded. This is achieved by
```python
import numpy as np
section_marker = np.zeros_like(x, dtype=int)
section_marker[1:12] = 1
```
leaving us with 10 samples. We would like to compute dissipation using half-overlapping FFT _segments_ of 2 samples, taking 3 such FFT _segments_ for each dissipation estimate (_chunk_), and an overlap of half a FFT _segment_ between _chunks_. Then:
```python
segment_len = 2
segment_overlap = 1
chunk_len = 4 # We can fit 3 half-overlapping segments in here
chunk_overlap = 1
```

### Implementation

In order to use numpy's vectorized routines for time series analysis, we use the function `get_chunking_index`. It takes in the parameters `segment_len`, `segment_overlap`, `chunk_len`, `chunk_overlap`, and `section_marker`, in addition to the length of the time series `samples_len`, and return an array of indices (see sketch), called `[ii]`. 

TODO find names for function_name and `ii`.

Index `ii`, when used to index an axis of length `samples_len` of any array, will trigger expansion of the time axis into three axes, that are, in turn: 
1. The slow/reduced/aggregated time axis, i.e. counting chunks.
2. Inside each chunk, counting the number of segments.
3. Inside each segment, counting the samples attributed to each segment.

For instance, given a an array `x` whose last axis has length `samples_len`, we can calculate the FFT over all segments without any loop:
```python
xr = x[..., ii] # reshape time axis. Now the last dimension contains the FFT segments
xr -= xr.mean(axis=-1)[..., newaxis]  # subtract mean
xr *= np.hanning(fft_length)[newaxis, newaxis, :]  # hanning window
Fx = np.fft.rfft(xr) # FFT(x). Now the last dimension contains the frequency axis
```

Similarly, one can easily calculate segment- or chunkwise variance, trends, or other. The same logic is also used for aggregating variables from "fast" to "slow" time.