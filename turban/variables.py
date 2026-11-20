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
    "chi/explanation": "Dissipation rate of temperature variance",
    "chi/latex": r"$\chi$",
    "chi/unit": "K2 s-1",
    "cond/long_name": "sea water electrical conductivity",
    "cond/standard_name": "sea_water_electrical_conductivity",
    # "cond/units": "S m-1", # TODO: not for MSS - how to override?
    "data_length/explanation": "Length of data used for one dissipation estimate",
    "data_length/latex": r"$l_\epsilon$",
    "data_length/units": "m",
    "dens/long_name": "sea water density",
    "dens/standard_name": "sea_water_density",
    "dens/units": "kg m-3",
    "eps/atomix_name": "EPSI",
    "eps/explanation": "Dissipation rate of turbulent kinetic energy per unit mass",
    "eps/latex": r"$\epsilon$",
    "eps/standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
    "eps/units": "W kg-1",
    "eps_final/standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
    "eps_final/long_name": "turbulent kinetic energy dissipation rate, final estimate",
    "eps_final/units": "W kg-1",
    "fom/atomix_name": "FOM",
    "fom/explanation": "Figure of merit: log_diss_mad divided by its expected upper 95% confidence bound",
    "fom/latex": r"$\mathrm{FOM}$",
    "fom/units": "-",
    # "freq/atomix_name": None,  # TODO
    "freq/explanation": "Number of cycles per second",
    "freq/latex": r"$f$",
    "freq/units": "Hz",  # cps
    "kolmlen/explanation": "Kolmogorov length scale",
    "kolmlen/latex": r"$\eta$",
    "kolmlen/units": "m",
    "molvisc/atomix_name": "KVISC",
    "molvisc/explanation": "Kinematic molecular viscosity of sea water",
    "molvisc/latex": r"$\nu$",
    "molvisc/long_name": "molecular_viscosity_of_seawater",
    "molvisc/unit": "m2 s-1",
    "moldiff/explanation": "Kinematic molecular diffusivity of sea water",
    "moldiff/latex": r"$\kappa$",
    "moldiff/long_name": "molecular_diffusivity_of_seawater",
    "molvisc/units": "m2 s-1",
    "press/explanation": "Sea water pressure",
    "press/long_name": "sea water pressure",
    "press/standard_name": "sea_water_pressure",
    "press/units": "dbar",
    "psi_f/explanation": "Generic power spectrum in units of Hz-1",
    "psi_k/explanation": "Generic power spectrum in units of cpm-1",
    "psi/explanation": "Generic power spectrum, in frequency or wavenumber domain",
    "sal/long_name": "sea water salinity",
    "sal/standard_name": "sea_water_salinity",
    "sal/units": "1e-3",
    "sampfreq/atomix_name": "fs",
    "sampfreq/explanation": "Number of samples per second",
    "sampfreq/units": "Hz",
    "section_number/atomix_name": "SECTION_NUMBER",
    "section_number/explanation": "Identifier of the continuous data section selected for dissipation estimates, 0 for discarded data",
    "section_number/units": "-",
    "senspeed/atomix_name": "PSPD_REL",
    "senspeed/explanation": "Speed of the sensor/platform relative to water",
    "senspeed/latex": r"$W$",
    "senspeed/units": "m s-1",
    "temp/long_name": "sea water temperature",
    "temp/standard_name": "sea_water_temperature",
    "temp/units": "degC",
    "time/atomix_name": "TIME",
    "time/explanation": "Time of the sample, e.g. as datetime or sample count",
    "waveno/atomix_name": "KCYC",
    "waveno/explanation": "Number of cycles per metre",
    "waveno/latex": r"$k$",
    "waveno/units": "cpm",
    "waveno_cutoff/atomix_name": "KMAX",
    "waveno_cutoff/explanation": "Upper wavenumber limit of the spectral values used for the dissipation estimate",
    "waveno_cutoff/units": "cpm",
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
