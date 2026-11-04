"""
Variables specific to microtemperature processing, in addition to those in
``turban.variables``. Same structure as there: ``_utemp_vars`` is the flat
``{varname/attribute: value}`` form of the nested ``VARIABLES_UTEMP``.
"""

from turban.variables import _to_dict # this function can vanish anytime

_utemp_vars = {
    "eps_lower/explanation": "TKE dissipation rate at the lower bound of the 95% confidence interval of the Batchelor wavenumber",
    "eps_lower/unit": "W kg-1",
    "eps_upper/explanation": "TKE dissipation rate at the upper bound of the 95% confidence interval of the Batchelor wavenumber",
    "eps_upper/unit": "W kg-1",
    "likelihood_ratio/explanation": "Likelihood of the fitted theoretical temperature gradient spectrum relative to that of a power law fit (not log10)",
    "likelihood_ratio/unit": "-",
    "mad_ratio/explanation": "Mean absolute deviation of the ratio of observed to fitted spectrum from its mean, over the fit range",
    "mad_ratio/unit": "-",
    "mad_powerlaw/explanation": "As mad_ratio, but for a power law fit to the observed spectrum",
    "mad_powerlaw/unit": "-",
    "kbatch_ci/explanation": "Half width of the 95% confidence interval of the Batchelor wavenumber",
    "kbatch_ci/unit": "cpm",
    "kbatch/explanation": "Batchelor wavenumber from the maximum likelihood fit",
    "kbatch/latex": r"$k_B$",
    "kbatch/unit": "cpm",
}

VARIABLES_UTEMP = _to_dict(_utemp_vars)
