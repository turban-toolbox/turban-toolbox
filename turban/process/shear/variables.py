"""
Variables specific to shear processing, in addition to those in
``turban.variables``. Same structure as there: ``_shear_vars`` is the flat
``{varname/attribute: value}`` form of the nested ``VARIABLES_SHEAR``.
"""

from turban.variables import _to_dict  # this function can vanish anytime

_shear_vars = {
    "eps_source_flag/explanation": "Method of dissipation estimate: 1 spectral integration, 2 fit to the inertial subrange",
    "eps_source_flag/units": "-",
    "freq_cutoff_antialias/atomix_name": "f_AA",
    "freq_cutoff_antialias/explanation": "Anti-aliasing filter frequency f_AA. It sets the candidate k_AA = 0.9 f_AA / W, with W the profiling speed, for the upper limit k_u of spectral integration (Eq. 3; Sec. 3.4.1 in the ATOMIX paper). 0.9 suits an 8th-order Butterworth filter",
    "freq_cutoff_antialias/latex": r"$k_{AA} = 0.9\,f_{AA}/W$",
    "freq_cutoff_antialias/units": "Hz",
    "freq_cutoff_corrupt/atomix_name": "f_limit",
    "freq_cutoff_corrupt/explanation": "Frequency f_lim of irremovable spectral corruption, usually at a fixed frequency (infinite for good data). It sets the candidate k_lim = f_lim / W, with W the profiling speed, for the upper limit k_u of spectral integration (Eq. 3; Sec. 3.4.1 in the ATOMIX paper)",
    "freq_cutoff_corrupt/latex": r"$k_{lim} = f_{lim}/W$",
    "freq_cutoff_corrupt/units": "Hz",
    "log_diss_mad/atomix_name": "MAD",
    "log_diss_mad/explanation": "Mean absolute deviation of the natural logarithm of the shear spectrum from that of the model spectrum, over the wavenumbers used for the dissipation estimate",
    "log_diss_mad/latex": r"$\mathrm{MAD}_{\ln\Psi}$",
    "log_diss_mad/units": "-",
    "log_diss_std/atomix_name": "EPSI_STD",
    "log_diss_std/explanation": "Standard deviation of the natural logarithm of the dissipation estimate",
    "log_diss_std/latex": r"$\sigma_{\ln\epsilon}$",
    "log_diss_std/units": "-",
    "log_psi_std/atomix_name": "SPEC_STD",
    "log_psi_std/explanation": "Standard deviation of the natural logarithm of the shear spectrum",
    "log_psi_std/latex": r"$\sigma_{\ln\Psi}$",
    "log_psi_std/units": "-",
    "max_despike_iter/atomix_name": "DESPIKE_PASS_COUNT_SH",
    "max_despike_iter/explanation": "Maximum number of despiking passes over the samples of the dissipation chunk",
    "max_despike_iter/units": "-",
    "num_despike_iter/explanation": "Number of despiking passes in which the sample was detected as part of a spike",
    "num_despike_iter/units": "-",
    "num_spec_points/atomix_name": "N_S",
    "num_spec_points/explanation": "Number of spectral values used for the dissipation estimate, excluding zero wavenumber",
    "num_spec_points/latex": r"$N_s$",
    "num_spec_points/units": "-",
    "number_signals_vibration_removal/atomix_name": "N_VIB_SENSORS",
    "number_signals_vibration_removal/explanation": "Number of vibration signals used to remove vibration-coherent noise from the shear spectra",
    "number_signals_vibration_removal/latex": r"$N_V$",
    "number_signals_vibration_removal/units": "-",
    "quality_metric/atomix_name": "EPSI_FLAGS",
    "quality_metric/explanation": "Binary-weighted quality flag, sum of failures: 1 figure of merit, 2 spike fraction, 4 disagreement between probes, 8 despiking passes, 16 resolved variance",
    "quality_metric/latex": r"$Q$",
    "quality_metric/units": "-",
    "resolved_var_frac/atomix_name": "VAR_RESOLVED",
    "resolved_var_frac/explanation": "Fraction of the shear variance resolved by spectral integration",
    "resolved_var_frac/latex": r"$V_f$",
    "resolved_var_frac/units": "-",
    "shear/atomix_name": "SHEAR",
    "shear/explanation": "Current shear",
    "shear/latex": r"$\partial u / \partial z$",
    "shear/units": "s-1",
    "spike_fraction/atomix_name": "DESPIKE_FRACTION_SH",
    "spike_fraction/explanation": "Fraction of samples in the dissipation chunk modified by despiking",
    "spike_fraction/units": "-",
    "waveno_cutoff_spatial_corr/explanation": "Wavenumber k_SR at which the spatial-response correction of the shear probe (Eq. 14 in the ATOMIX paper) reaches 10, about 150 cpm for standard probes. Soft candidate for the upper limit k_u of spectral integration (Eq. 3; Sec. 3.4.1 in the ATOMIX paper)",
    "waveno_cutoff_spatial_corr/latex": r"$k_{SR}$",
    "waveno_cutoff_spatial_corr/units": "cpm",
    "waveno_spectral_min/explanation": "Wavenumber k_min of the spectral minimum between the viscous roll-off and the rise of electronic noise, which the ATOMIX paper finds by a polynomial fit in log-log space and does not accept below 10 cpm. Here a fixed value set by the user. Candidate for the upper limit k_u of spectral integration (Eq. 3; Sec. 3.4.1 in the ATOMIX paper)",
    "waveno_spectral_min/latex": r"$k_{min}$",
    "waveno_spectral_min/units": "cpm",
    # "eps_source_flag/atomix_name": "METHOD",  # METHOD = eps_source_flag - 1
    # "waveno_cutoff_spatial_corr/atomix_name": None,  # no counterpart in the ATOMIX benchmark netCDF files
    # "waveno_spectral_min/atomix_name": None,  # ATOMIX files store only the polynomial order (fit_order), not k_min
}

VARIABLES_SHEAR = _to_dict(_shear_vars)
