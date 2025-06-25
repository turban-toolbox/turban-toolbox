from collections import defaultdict

# Physical quantities used in the code base
_vars = {
    "cond/standard_name": "sea_water_conductivity",
    "eps/atomix_name": "EPSI",
    "eps/latex": r"$\eps$",
    "eps/standard_name": "",
    "eps/unit": "W kg-1",
    "freq/atomix_name": None,
    "freq/explanation": "Number of cycles per second",
    "freq/unit": "Hz",  # cps
    "kolm_length/explanation": "Kolmogorov length scale",
    "kolm_length/latex": r"$\eta$",
    "molvisc/standard_name": "",  # TODO
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
    "sal/standard_name": "sea_water_salinity",  # TODO: absolute?
    "sampling_freq/atomix_name": "fs",
    "sampling_freq/explanation": "Number of samples per second",
    "sampling_freq/unit": "Hz",
    "senspeed/atomix_name": "PSPD",
    "senspeed/explanation": "Speed of the sensor relative to water",
    "shear/atomix_name": "SH",
    "shear/explanation": "Current shear",
    "shear/unit": "s-1",
    "temp/standard_name": "sea_water_temperature",
    "temp/unit": "degC",
    "wavno/atomix_name": "KCYC",  # TODO
    "wavno/explanation": "Number of cycles per metre",
    "wavno/unit": "cpm",
    "x/explanation": "Generic signal",
}


def to_dict(vars):
    dct = defaultdict(lambda: {})
    for k, v in vars.items():
        varname, attr = k.split("/")
        dct[varname][attr] = k
    return dct


VARIABLES = to_dict(_vars)
