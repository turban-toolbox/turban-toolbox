"""
Variables specific to shear processing, in addition to those in
``turban.variables``. Same structure as there: ``_shear_vars`` is the flat
``{varname/attribute: value}`` form of the nested ``VARIABLES_SHEAR``.
"""

from turban.variables import _to_dict  # this function can vanish anytime

_shear_vars = {
    "freq_cutoff_antialias/atomix_name": "f_AA",
    "freq_cutoff_antialias/explanation": "Anti-aliasing filter frequency f_AA. It sets the candidate k_AA = 0.9 f_AA / W, with W the profiling speed, for the upper limit k_u of spectral integration (Eq. 3; Sec. 3.4.1 in the ATOMIX paper). 0.9 suits an 8th-order Butterworth filter",
    "freq_cutoff_antialias/latex": r"$k_{AA} = 0.9\,f_{AA}/W$",
    "freq_cutoff_antialias/unit": "Hz",
    "freq_cutoff_corrupt/atomix_name": "f_limit",
    "freq_cutoff_corrupt/explanation": "Frequency f_lim of irremovable spectral corruption, usually at a fixed frequency (infinite for good data). It sets the candidate k_lim = f_lim / W, with W the profiling speed, for the upper limit k_u of spectral integration (Eq. 3; Sec. 3.4.1 in the ATOMIX paper)",
    "freq_cutoff_corrupt/latex": r"$k_{lim} = f_{lim}/W$",
    "freq_cutoff_corrupt/unit": "Hz",
    # "waveno_cutoff_spatial_corr/atomix_name": None,  # no counterpart in the ATOMIX benchmark netCDF files
    "waveno_cutoff_spatial_corr/explanation": "Wavenumber k_SR at which the spatial-response correction of the shear probe (Eq. 14 in the ATOMIX paper) reaches 10, about 150 cpm for standard probes. Soft candidate for the upper limit k_u of spectral integration (Eq. 3; Sec. 3.4.1 in the ATOMIX paper)",
    "waveno_cutoff_spatial_corr/latex": r"$k_{SR}$",
    "waveno_cutoff_spatial_corr/unit": "cpm",
    # "waveno_spectral_min/atomix_name": None,  # ATOMIX files store only the polynomial order (fit_order), not k_min
    "waveno_spectral_min/explanation": "Wavenumber k_min of the spectral minimum between the viscous roll-off and the rise of electronic noise, which the ATOMIX paper finds by a polynomial fit in log-log space and does not accept below 10 cpm. Here a fixed value set by the user. Candidate for the upper limit k_u of spectral integration (Eq. 3; Sec. 3.4.1 in the ATOMIX paper)",
    "waveno_spectral_min/latex": r"$k_{min}$",
    "waveno_spectral_min/unit": "cpm",
}

VARIABLES_SHEAR = _to_dict(_shear_vars)
