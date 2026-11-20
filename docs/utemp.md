# Microtemperature (chi) processing

The microtemperature pipeline estimates the dissipation rate of temperature variance $\chi$ and, through the Batchelor wavenumber, the dissipation rate of turbulent kinetic energy $\epsilon$ from fast thermistor (FP07) data.

!!! warning "Experimental"
	The pipeline runs but is still under testing and development. Level 2 does not despike yet, and the fit at Level 4 is still being refined. Interfaces and results may change.

## Overview

The pipeline is structured like the [shear pipeline](manual.md), with the classes in `turban.process.utemp.api`:

| Level | Class | Content |
|---|---|---|
| 1 | `UTempLevel1` | Time derivative of temperature `dtempdt` (K s⁻¹), sensor speed, section numbers |
| 2 | `UTempLevel2` | Cleaned `dtempdt` (currently a copy of Level 1) |
| 3 | `UTempLevel3` | Chunk-wise temperature gradient spectra and noise spectra |
| 4 | `UTempLevel4` | $\chi$, $\epsilon$ and quality metrics from a maximum likelihood fit of a theoretical spectrum |

`UTempProcessing` chains the levels, and all levels support the same export and import to xarray and netCDF as the shear levels (see [Exporting and importing data](manual.md#exporting-and-importing-data)). Note that, unlike the shear pipeline, Level 1 holds the *time* derivative $\partial T/\partial t$; it is converted to the spatial gradient at Level 3.

## Example

The following runs the pipeline on synthetic data. For real data from the MSS, see [Microtemperature from MSS data](mss.md#microtemperature).

```python
import numpy as np
from turban.process.utemp.api import UTempLevel1, UTempProcessing
from turban.process.utemp.config import UTempConfig

sampfreq = 512.0
n = int(20 * sampfreq)
rng = np.random.default_rng(0)

cfg = UTempConfig(
    sampfreq=sampfreq,
    segment_length=256,
    segment_overlap=128,
    chunk_length=1024,
    chunk_overlap=512,
    fp07_response="nash99",  # see FP07 response below
    spectrum_model="batchelor",
)
level1 = UTempLevel1(
    time=np.arange(n) / sampfreq,
    senspeed=0.7 * np.ones(n),  # [m s-1]
    dtempdt=0.01 * rng.standard_normal((2, n)),  # [K s-1], two thermistors
    section_number=np.ones(n, dtype=int),
    cfg=cfg,
)
# optional: in-situ temperature for viscosity and diffusivity
level1.add_aux_data(np.full(n, 8.0), "temp", "mean", "temp")

p = UTempProcessing(level1)
ds4 = p.level4.to_xarray()  # chi, eps, kbatch, ...
data_tree = p.to_xarray()  # all levels
```

## Configuration

`UTempConfig` takes the segment and chunk parameters of `SegmentConfig` (see [Sections, segments, and chunks](manual.md#sections-segments-and-chunks)) and the following:

| Parameter | Default | Used at | Meaning |
|---|---|---|---|
| `fp07_response` | `"nash99"` | Level 3 | Frequency response of the thermistor, see below |
| `molvisc_fallback` | `1.6e-6` | Level 4 | Kinematic viscosity [m² s⁻¹] if there is no aux variable `temp` |
| `moldiff_fallback` | `1.4e-7` | Level 4 | Thermal diffusivity [m² s⁻¹] if there is no aux variable `temp` |
| `spectrum_model` | `"batchelor"` | Level 4 | Theoretical spectrum: `"batchelor"` (Batchelor 1959) or `"kraichnan"` (Kraichnan 1968) |
| `q_batchelor` | `3.7` | Level 4 | Universal constant $q$ of the Batchelor spectrum (Oakey 1982) |
| `q_kraichnan` | `5.26` | Level 4 | Universal constant $q$ of the Kraichnan spectrum (Bogucki et al. 1997) |
| `waveno_limit_upper` | `None` | Level 4 | Hard upper limit [cpm] of the fit range, `None` for none |

The read-only properties `q` (the $q$ of the selected `spectrum_model`) and `dof_spectrum` (degrees of freedom of the spectral estimates, $1.9 N_f$ with $N_f$ the number of FFT segments per chunk, Nuttall 1971) are derived from these.

## FP07 response

The FP07 thermistor attenuates high frequencies. TURBAN models its power transfer function as

$$H(f) = \left[1 + (2\pi\tau f)^2\right]^{-n}, \qquad \tau = \tau_0 \left(\frac{W}{1\,\mathrm{m\,s^{-1}}}\right)^{\gamma},$$

with $n$ poles and a time constant $\tau$ depending on the sensor speed $W$. Level 3 divides the spectra by $H$. The following presets are available in `turban.process.utemp.config.FP07_RESPONSE_MODELS`:

| Name | $\tau_0$ [ms] | $\gamma$ | $n$ | Reference |
|---|---|---|---|---|
| `nash99` (default) | 7 | −0.12 | 1 | Nash et al. (1999), as in Piccolroaz et al. (2021) |
| `rsi` | 4.1 | −0.5 | 2 | Vachon & Lueck (1984), RSI default |
| `gm` | 10.6 | −0.32 | 2 | Gregg & Meagher (1980) |
| `kocsis` | 7 | −0.32 | 2 | Kocsis et al. (1999) |
| `nash02` | 8.5 | 0 | 2 | Nash et al. (2002), according to Sommer et al. (2013) |
| `sommer` | 10 | 0 | 2 | Sommer et al. (2013) |
| `ijichi25` | 5 | −0.5 | 1 | Ijichi & St. Laurent (2025) |

`fp07_response` accepts the name of a preset, a custom `FP07Response`, or `None` (no correction):

```python
from turban.process.utemp.config import FP07Response, UTempConfig

segment_kwargs = dict(
    sampfreq=1024.0,
    segment_length=2048,
    segment_overlap=1024,
    chunk_length=4 * 2048,
    chunk_overlap=1024,
)
cfg_preset = UTempConfig(**segment_kwargs, fp07_response="rsi")
cfg_custom = UTempConfig(
    **segment_kwargs,
    fp07_response=FP07Response(name="custom", tau0=6e-3, gamma=-0.3, npoles=2),
)
cfg_none = UTempConfig(**segment_kwargs, fp07_response=None)
```

The response is defined by `tau0`, `gamma` and `npoles` only; `name` is informative.

## Spectral correction

Instrument electronics may further modify the measured spectra, e.g. through the pre-emphasis (differentiator) of the thermistor channel and its digital removal. `UTempLevel1.spectral_correction` takes a multiplicative correction of the power spectra, evaluated on the frequency grid of the FFT segments, i.e. `np.fft.rfftfreq(cfg.segment_length, 1 / cfg.sampfreq)`. A warning is logged if its length does not match.

For a pre-emphasised channel deconvolved with a digital first-order low-pass (as for the MSS), `turban.process.utemp.util.correction_frequency_response_bilinear(freq, sampfreq, diff_gain)` returns the ratio of the ideal analog to the digital response. The MSS loader sets this correction automatically.

## Level 3: spectra and noise

For each chunk, Level 3 computes

- `psi_f_gradt`: the frequency spectrum of `dtempdt`, multiplied by `spectral_correction`,
- `psi_k_gradt`: the wavenumber spectrum of the temperature gradient, $\Psi_k = \Psi_f / (W H)$, using the chunk-averaged sensor speed $W$ and the FP07 response $H$,
- `psi_noise`: the noise spectrum, in the same units as `psi_k_gradt`,
- `waveno` = `freq` / $W$ for each chunk.

The noise is estimated empirically, separately for each thermistor: it is the (logarithmic) mean of the 5% least intense frequency spectra, where the electronic noise is the same for every chunk. It is then converted to the wavenumber domain with the sensor speed and FP07 response of each chunk, like `psi_k_gradt`.

## Level 4: maximum likelihood fit

Level 4 fits the theoretical temperature gradient spectrum to `psi_k_gradt` by maximum likelihood (Ruddick et al. 2000; Steinbuck et al. 2009). The implementation is a port of `MLE_Fit_Kb.m`, separately for each thermistor and chunk:

1. The upper limit $k_u$ of the fit range is the first wavenumber where the spectrum falls to twice the noise or below, at most `waveno_limit_upper`. With fewer than 9 spectral values above twice the noise, all Level 4 results are NaN.
2. A first estimate $\chi = 6\kappa \int \Psi_k\, dk$ is integrated up to $k_u$.
3. A coarse search over 100 trial Batchelor wavenumbers $k_B$ maximises the likelihood of the observed spectrum (assuming $\chi^2$-distributed spectral estimates with `dof_spectrum` degrees of freedom) given the theoretical spectrum plus noise.
4. $\chi$ is corrected for the variance outside the resolved range, using the theoretical spectrum of the coarse $k_B$.
5. A fine search around the coarse estimate gives $k_B$, and the curvature of the likelihood its 95% confidence interval. $\chi$ is recomputed from the fitted spectrum.
6. $\epsilon = \nu\kappa^2(2\pi k_B)^4$, with $k_B$ in cpm (Batchelor 1959).

Viscosity $\nu$ and diffusivity $\kappa$ are computed from an auxiliary variable `temp` if present, otherwise `molvisc_fallback` and `moldiff_fallback` are used, see [Molecular properties of seawater](manual.md#molecular-properties-of-seawater).

`UTempLevel4` contains:

| Variable | Meaning |
|---|---|
| `chi` | Dissipation rate of temperature variance [K² s⁻¹] |
| `eps` | Dissipation rate of turbulent kinetic energy [W kg⁻¹] |
| `eps_lower`, `eps_upper` | $\epsilon$ at the bounds of the 95% confidence interval of $k_B$ |
| `kbatch` | Batchelor wavenumber $k_B$ [cpm] |
| `kbatch_ci` | Half width of the 95% confidence interval of $k_B$ [cpm] |
| `likelihood_ratio` | Likelihood of the fitted theoretical spectrum relative to that of a power law fit (not log10) |
| `mad_ratio` | Mean absolute deviation of the ratio of observed to fitted spectrum (plus noise) |
| `mad_ratio_pow` | Mean absolute deviation of the ratio of observed to power law spectrum |
| `molvisc`, `moldiff` | Viscosity and diffusivity used [m² s⁻¹] |

A large `likelihood_ratio` and a small `mad_ratio` indicate that the theoretical spectrum describes the data well. TURBAN does not yet combine these into a quality flag.