"""
Defines standard names of variables used in the code base. Code should deviate from
these names only with good reason.

Dictionary `VARIABLES`` is a nested structure of the form:
{turban_standard_name: {attribute_name: attribute_value}}.

To make alphabetic sorting easier, this is here coded as the simpler dictionary `_vars`:
{turban_standard_name/attribute_name: attribute_value}
"""

from typing import Literal
from collections import defaultdict

# Physical quantities used in the code base
_vars = {
    "cond/standard_name": "sea_water_conductivity",
    "data_length/explanation": "Length of data used for one dissipation estimate",
    "data_length/latex": r"$l_\epsilon$",
    "data_length/unit": "m",
    "eps/atomix_name": "EPSI",
    "eps/latex": r"$\epsilon$",
    "eps/standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
    "eps/unit": "W kg-1",
    # "eps_source_flag/atomix_name": "METHOD",  # METHOD = eps_source_flag - 1
    "eps_source_flag/explanation": "Method of dissipation estimate: 1 spectral integration, 2 fit to the inertial subrange",
    "eps_source_flag/unit": "-",
    "fom/atomix_name": "FOM",
    "fom/explanation": "Figure of merit: log_diss_mad divided by its expected upper 95% confidence bound",
    "fom/latex": r"$\mathrm{FOM}$",
    "fom/unit": "-",
    # "freq/atomix_name": None,  # TODO
    "freq/explanation": "Number of cycles per second",
    "freq/latex": r"$f$",
    "freq/unit": "Hz",  # cps
    "kolmlen/explanation": "Kolmogorov length scale",
    "kolmlen/latex": r"$\eta$",
    "kolmlen/unit": "m",
    "log_diss_mad/atomix_name": "MAD",
    "log_diss_mad/explanation": "Mean absolute deviation of the natural logarithm of the shear spectrum from that of the model spectrum, over the wavenumbers used for the dissipation estimate",
    "log_diss_mad/latex": r"$\mathrm{MAD}_{\ln\Psi}$",
    "log_diss_mad/unit": "-",
    "log_diss_std/atomix_name": "EPSI_STD",
    "log_diss_std/explanation": "Standard deviation of the natural logarithm of the dissipation estimate",
    "log_diss_std/latex": r"$\sigma_{\ln\epsilon}$",
    "log_diss_std/unit": "-",
    "log_psi_std/atomix_name": "SPEC_STD",
    "log_psi_std/explanation": "Standard deviation of the natural logarithm of the shear spectrum",
    "log_psi_std/latex": r"$\sigma_{\ln\Psi}$",
    "log_psi_std/unit": "-",
    "max_despike_iter/atomix_name": "DESPIKE_PASS_COUNT_SH",
    "max_despike_iter/explanation": "Maximum number of despiking passes over the samples of the dissipation chunk",
    "max_despike_iter/unit": "-",
    "molvisc/atomix_name": "KVISC",
    "molvisc/explanation": "Kinematic molecular viscosity of sea water",
    "molvisc/latex": r"$\nu$",
    "molvisc/long_name": "molecular_viscosity_of_seawater",
    "molvisc/unit": "m2 s-1",
    "num_despike_iter/explanation": "Number of despiking passes in which the sample was detected as part of a spike",
    "num_despike_iter/unit": "-",
    "num_spec_points/atomix_name": "N_S",
    "num_spec_points/explanation": "Number of spectral values used for the dissipation estimate, excluding zero wavenumber",
    "num_spec_points/latex": r"$N_s$",
    "num_spec_points/unit": "-",
    "number_signals_vibration_removal/atomix_name": "N_VIB_SENSORS",
    "number_signals_vibration_removal/explanation": "Number of vibration signals used to remove vibration-coherent noise from the shear spectra",
    "number_signals_vibration_removal/latex": r"$N_V$",
    "number_signals_vibration_removal/unit": "-",
    "press/explanation": "Sea water pressure",
    "press/standard_name": "sea_water_pressure",
    "psi_f_gradt/explanation": "Power density spectrum of temperature gradient variance w.r.t frequency",
    "psi_f_gradt/latex": r"$\Psi_{Tz}$",
    "psi_f_gradt/long_name": "",
    "psi_f_gradt/unit": "K2 m-2 Hz-1",
    "psi_f_sh/explanation": "Power density spectrum of shear variance w.r.t frequency",
    "psi_f_sh/long_name": "",
    "psi_f_sh/unit": "s-2 Hz-1",
    "psi_f/explanation": "Generic power spectrum in units of Hz-1",
    "psi_k_gradt/explanation": "Power density spectrum of temperature gradient variance w.r.t wavenumber",
    "psi_k_gradt/latex": r"$\Psi_{Tz}$",
    "psi_k_gradt/long_name": "",
    "psi_k_gradt/unit": "K2 m-2 cpm-1",
    # "psi_k_sh/atomix_name": "SH_SPEC" | "SH_SPEC_CLEAN",
    "psi_k_sh/explanation": "Power density spectrum of shear variance w.r.t wavenumber",
    "psi_k_sh/latex": r"$\Psi(k)$",
    "psi_k_sh/long_name": "",
    "psi_k_sh/unit": "s-2 cpm-1",
    "psi_k_sh_confidence_interval/explanation": "Upper and lower bound of the 95% confidence interval of psi_k_sh",
    "psi_k_sh_confidence_interval/latex": r"$\Psi(k)\exp(\pm 1.96\,\sigma_{\ln\Psi})$",
    "psi_k_sh_confidence_interval/unit": "s-2 cpm-1",
    "psi_k/explanation": "Generic power spectrum in units of cpm-1",
    "psi/explanation": "Generic power spectrum, in frequency or wavenumber domain",
    "quality_metric/atomix_name": "EPSI_FLAGS",
    "quality_metric/explanation": "Binary-weighted quality flag, sum of failures: 1 figure of merit, 2 spike fraction, 4 disagreement between probes, 8 despiking passes, 16 resolved variance",
    "quality_metric/latex": r"$Q$",
    "quality_metric/unit": "-",
    "resolved_var_frac/atomix_name": "VAR_RESOLVED",
    "resolved_var_frac/explanation": "Fraction of the shear variance resolved by spectral integration",
    "resolved_var_frac/latex": r"$V_f$",
    "resolved_var_frac/unit": "-",
    "sal/standard_name": "sea_water_salinity",
    "sampfreq/atomix_name": "fs",
    "sampfreq/explanation": "Number of samples per second",
    "sampfreq/unit": "Hz",
    "section_number/atomix_name": "SECTION_NUMBER",
    "section_number/explanation": "Identifier of the continuous data section selected for dissipation estimates, 0 for discarded data",
    "section_number/unit": "-",
    "senspeed/atomix_name": "PSPD_REL",
    "senspeed/explanation": "Speed of the sensor/platform relative to water",
    "senspeed/latex": r"$W$",
    "senspeed/unit": "m s-1",
    "shear/atomix_name": "SHEAR",
    "shear/explanation": "Current shear",
    "shear/latex": r"$\partial u / \partial z$",
    "shear/unit": "s-1",
    "spike_fraction/atomix_name": "DESPIKE_FRACTION_SH",
    "spike_fraction/explanation": "Fraction of samples in the dissipation chunk modified by despiking",
    "spike_fraction/unit": "-",
    "temp/standard_name": "sea_water_temperature",
    "temp/unit": "degC",
    "time/atomix_name": "TIME",
    "time/explanation": "Time of the sample, e.g. as datetime or sample count",
    "waveno/atomix_name": "KCYC",
    "waveno/explanation": "Number of cycles per metre",
    "waveno/latex": r"$k$",
    "waveno/unit": "cpm",
    "waveno_cutoff/atomix_name": "KMAX",
    "waveno_cutoff/explanation": "Upper wavenumber limit of the spectral values used for the dissipation estimate",
    "waveno_cutoff/unit": "cpm",
    "x/explanation": "Generic signal",
}

def _to_dict(vars):
    """Transform a flat dictionary into a nested variable-attribute structure.

    This function may vanish at any time - its only purpose is converting _vars
    to VARIABLES.

    Parameters
    ----------
    vars : dict
        Flat dictionary with keys formatted as ``"varname/attribute"``.

    Returns
    -------
    dict
        Nested dictionary mapping variable names to attribute dictionaries.
        Default factory returns an empty dict for missing keys.
    """
    dct = defaultdict(lambda: {})
    for k, v in vars.items():
        varname, attr = k.split("/")
        dct[varname][attr] = v
    return dct


VARIABLES = _to_dict(_vars)

# Standard names of CTD-type auxiliary variables an instrument sensor can be mapped to
# (e.g. `MssDeviceConfig.sensornames_ctd`). They specifically denote:
#   cond: sea water conductivity
#   press: sea water pressure
#   temp: in-situ sea water temperature (ITS-90)
STANDARD_CTD_TYPE = Literal["cond", "press", "temp"]
