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
    "cond/standard_name": "sea_water_conductivity",
    "eps/atomix_name": "EPSI",
    "eps/latex": r"$\eps$",
    "eps/standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
    "eps/unit": "W kg-1",
    "freq/atomix_name": None,
    "freq/explanation": "Number of cycles per second",
    "freq/unit": "Hz",  # cps
    "kolmlen/explanation": "Kolmogorov length scale",
    "kolmlen/latex": r"$\eta$",
    "molvisc/long_name": "molecular_viscosity_of_seawater",
    "molvisc/unit": "m2 s-1",
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
    "sal/standard_name": "sea_water_salinity",
    "sampfreq/atomix_name": "fs",
    "sampfreq/explanation": "Number of samples per second",
    "sampfreq/unit": "Hz",
    "senspeed/atomix_name": "PSPD",
    "senspeed/explanation": "Speed of the sensor relative to water",
    "shear/atomix_name": "SH",
    "shear/explanation": "Current shear",
    "shear/unit": "s-1",
    "temp/standard_name": "sea_water_temperature",
    "temp/unit": "degC",
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
