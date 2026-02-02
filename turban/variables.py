"""
Defines standard names of variables used in the code base. Code should deviate from
these names only with good reason.

Dictionary `VARIABLES`` is a nested structure of the form:
{turban_standard_name: {attribute_name: attribute_value}}.

To make alphabetic sorting easier, this is here coded as the simpler dictionary `_vars`:
{turban_standard_name/attribute_name: attribute_value}
"""

from collections import defaultdict

# Physical quantities used in the code base
_vars = {
    "chunk_length/explanation": "Number of samples in a chunk (e.g. for estimating a spectrum)",
    "chunk_overlap/explanation": "Overlap (in samples) between consecutive chunks",
    "cond/standard_name": "sea_water_conductivity",
    "cutoff_freq_lp/explanation": "",
    "cutoff_freq_lp/unit": "Hz",
    "eps_source_flag/explanation": "Algorithm used for this eps estimate. 1: Spectral integration, 2: Inertial subrange fit.",
    "eps/atomix_name": "EPSI",
    "eps/latex": r"$\eps$",
    "eps/standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
    "eps/unit": "W kg-1",
    "freq_cutoff_antialias/explanation": "",
    "freq_cutoff_antialias/unit": "Hz",
    "freq_cutoff_corrupt/explanation": "",
    "freq_cutoff_corrupt/unit": "Hz",
    "freq_highpass/explanation": "",
    "freq_highpass/unit": "Hz",
    "freq/atomix_name": "", # TODO
    "freq/explanation": "Number of cycles per second",
    "freq/unit": "Hz",  # cps
    "kolmlen/explanation": "Kolmogorov length scale",
    "kolmlen/latex": r"$\eta$",
    "log_diss_mad/explanation": "log(eps) maximum absolute deviation",
    "log_diss_var/explanation": "log(eps) variance",
    "max_despike_iter/explanation": "Maximum number of times despiking has been applied to any of the samples in this chunk.",
    "max_tries/explanation": "Maximum number of despiking iterations on a given sample",
    "molvisc/long_name": "molecular_viscosity_of_seawater",
    "molvisc/unit": "m2 s-1",
    "num_despike_iter/explanation": "Number of times the despiking algorithm has been applied to this sample.",
    "num_spec_points/explanation": "Number of spectral points",
    "press/explanation": "Sea water pressure",
    "press/standard_name": "sea_water_pressure",
    "psi_f_gradt/explanation": "Power density spectrum of temperature gradient variance w.r.t frequency",
    "psi_f_gradt/latex": r"\Psi_{Tz}",
    "psi_f_gradt/long_name": "",
    "psi_f_gradt/unit": "K2 m-2 Hz-1",
    "psi_f_sh/atomix_name": "SPEC",
    "psi_f_sh/explanation": "Power density spectrum of shear variance w.r.t frequency",
    "psi_f_sh/long_name": "",
    "psi_f_sh/unit": "s-2 Hz-1",
    "psi_f/explanation": "Generic power spectrum in units of Hz-1",
    "psi_k_gradt/explanation": "Power density spectrum of temperature gradient variance w.r.t wavenumber",
    "psi_k_gradt/latex": r"\Psi_{Tz}",
    "psi_k_gradt/long_name": "",
    "psi_k_gradt/unit": "K2 m-2 cpm-1",
    "psi_k_sh/atomix_name": "SPEC",
    "psi_k_sh/explanation": "Power density spectrum of shear variance w.r.t wavenumber",
    "psi_k_sh/long_name": "",
    "psi_k_sh/unit": "s-2 cpm-1",
    "psi_k/explanation": "Generic power spectrum in units of cpm-1",
    "psi/explanation": "Generic power spectrum, in frequency or wavenumber domain",
    "quality_metric/explanation": "Integral number; sum of binary quality flags.",
    "resolved_var_frac/explanation": "Fraction of total variance resolved by spectrum",
    "sal/standard_name": "sea_water_salinity",
    "sampfreq/atomix_name": "fs",
    "sampfreq/explanation": "Sampling frequency, number of samples per second",
    "sampfreq/unit": "Hz",
    "section_number/explanation": "Integer number of contiguous sections of data. 0 means discard, >=1 denote sections to be analyzed",
    "segment_length/explanation": "Number of samples in a segment inside a chunk (e.g. for FFT)",
    "segment_overlap/explanation": "Overlap (in samples) between consecutive segments",
    "senspeed/atomix_name": "PSPD",
    "senspeed/explanation": "Speed of the sensor relative to water",
    "senspeed/long_name": "Platform speed through water",
    "senspeed/unit": "m s-1",
    "shear/atomix_name": "SH",
    "shear/explanation": "Current shear",
    "shear/unit": "s-1",
    "spatial_response_wavenum/explanation": "",
    "spatial_response_wavenum/unit": "m-1",
    "spike_fraction/explanation": "Fraction of despiked samples in this chunk.",
    "spike_include_after/explanation": "Number of samples included in a given spike after it",
    "spike_include_before/explanation": "Number of samples included in a given spike before it",
    "spike_replace_after/explanation": "Number of samples after a detected spike used to define value for replacing spike",
    "spike_replace_before/explanation": "Number of samples before a detected spike used to define value for replacing spike",
    "spike_threshold/explanation": "",
    "temp/standard_name": "sea_water_temperature",
    "temp/unit": "degC",
    "waveno_cutoff_spatial_corr/explanation": "",
    "waveno_cutoff_spatial_corr/unit": "m-1",
    "waveno/atomix_name": "KCYC",
    "waveno/explanation": "Number of cycles per metre",
    "waveno/unit": "cpm",
    "x/explanation": "Generic signal",
}


def to_dict(vars):
    dct = defaultdict(lambda: {})
    for k, v in vars.items():
        varname, attr = k.split("/")
        dct[varname][attr] = v
    return dct


VARIABLES = to_dict(_vars)
