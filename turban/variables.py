import json
import pydantic
from collections import defaultdict

FIXED_NAMES = { # TODO into respective instruments module?
    "TEMP_FAST": "Fast-response temperature",
    "TEMP_EMPH": "Fast-response temperature, pre-emphasized",
    "TEMP": "Precision temperature",
    "Cmdc": "fast response conductivity (low-pass filter to get conductivity)",
    "Cmac": "fluctuation from fast response conductivity (gradient gives mC gradients)",
    "COND": "Precision conductivity",
}

_generic = {
    "psi_f/explanation": "Generic power spectrum in units of Hz-1",
    "psi_k/explanation": "Generic power spectrum in units of cpm-1",
    "x/explanation": "Generic signal",
}

# Physical quantities used in the code base
_vars = {
    "eps/atomix_name": "EPSI",
    "eps/latex": r"$\eps$",
    "eps/standard_name": "",
    "eps/unit": "W kg-1",
    "freq/atomix_name": None,
    "freq/explanation": "Number of cycles per second",
    "freq/unit": "Hz",  # cps
    "psi_k_gradt/explanation": "Power density spectrum of temperature gradient variance",
    "psi_k_gradt/long_name": "",
    "psi_k_gradt/latex": r"\Psi_{Tz}",
    "psi_k_gradt/unit": "K2 m-2 cpm-1",
    "psi_sh/atomix_name": "SPEC",
    "psi_sh/explanation": "Power density spectrum of shear variance",
    "psi_sh/long_name": "",
    "psi_sh/unit": "s-2 cpm-1",
    "sampling_freq/atomix_name": "fs",
    "sampling_freq/explanation": "Number of samples per second",
    "sampling_freq/unit": "Hz",
    "sh/atomix_name": "SH",
    "temp/": "",
    "temp/standard_name": "sea_water_temperature",
    "temp/unit": "degC",
    "waveno/atomix_name": "KCYC",
    "waveno/unit": "cpm",
    # "waveno/ex": "",
}


def to_dict(vars):
    dct = defaultdict(lambda: {})
    for k, v in vars.items():
        varname, attr = k.split("/")
        dct[varname][attr] = k
    return dct


VARIABLES = to_dict(_vars)
GENERIC = to_dict(_generic)
