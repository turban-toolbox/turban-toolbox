import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray, newaxis
from scipy.signal import butter, freqz, lfilter, lfiltic
from scipy.special import erf, gamma

from turban.utils.util import integrate, reshape_any_first, reshape_halfoverlap_last

# nu, kin. viscosity of water; assumed known constant
viscosity_kinematic = 0.0000016
# molecular temperature diffusivity [m^2/s]
diffusivity_temp = 0.00000014
# constant for batchelor spectrum
q_b = 3.7


def _flatten01(x: ndarray):
    """
    Flatten first two dimensions of array
    """
    return x.reshape((x.shape[0] * x.shape[1], x.shape[2]))


def _try_sigma_derivative(df):
    """
    Cheap try to get sigma0 derivative from high precision cond/temp"""
    from functools import partial

    import gsw
    from level1 import get_vsink
    from turban.utils.util import butterfilt, fft_grad

    def derivative(func, arg, x, dx):
        return (func(**{arg: x + dx}) - func(**{arg: x - dx})) / 2 / dx

    def sigma0_from_C(C, t, p):
        SP = gsw.SP_from_C(C, t, p)
        SA = gsw.SA_from_SP(SP, p, 20, 80)  # lon/lat...
        CT = gsw.CT_from_t(SA, t, p)
        sigma = gsw.sigma0(SA, CT)
        return sigma

    df["dsigmadC"] = np.array(
        [
            derivative(partial(sigma0_from_C, t=t, p=p), "C", C, 0.01)
            for C, t, p in zip(df.Cond, df.Temp, df.Press)
        ]
    )
    df["dsigmadT"] = np.array(
        [
            derivative(partial(sigma0_from_C, C=C, p=p), "t", t, 0.01)
            for C, t, p in zip(df.Cond, df.Temp, df.Press)
        ]
    )

    senspeed = get_vsink(data_arrays["PRESSURE"], 1024)[0]
    df["dCdt"] = fft_grad(df.Cmac, 1 / 1024)
    df["dCdt"] = butterfilt(df.dCdt, 50, 1024)
    df["dTdt"] = fft_grad(df.NTCHP, 1 / 1024)
    df["dsigmadt"] = df["dCdt"] * df["dsigmadC"] + df["dTdt"] * df["dsigmadT"]
    df["dsigmadz"] = df.dsigmadt / senspeed


def _k_batchelor_mle_round_2():
    """# MLE fitting, round 2
    # This may give
    # TODO: speed up
    range_ = kmax_1 - kmin_1
    allrows = range(cfunc_values_1.shape[0])
    # cost function values to the left, right, and at its maximum value
    _left = cfunc_values_1[allrows, np.maximum(k_batchelor_idx_1 - 1, 0)]
    _right = cfunc_values_1[
        allrows, np.minimum(k_batchelor_idx_1 + 1, cfunc_values_1.shape[0] - 1)
    ]
    _mid = cfunc_values_1[allrows, k_batchelor_idx_1]
    assert np.max(cfunc_values_1[0]) == _mid[0]
    # curvature of cost function values at batchelor waveno
    delta_k = (2 * range_) / np.sqrt(2 * _mid - _left - _right)
    print(delta_k)
    finalrange = np.maximum(delta_k, range_)
    # print(finalrange)
    kmin_2: Float[ndarray, "time_slow"] = np.maximum(
        0, k_batchelor_estimates_1 - finalrange
    )
    kmax_2: Float[ndarray, "time_slow"] = k_batchelor_estimates_1 + finalrange
    print(kmin_2, kmax_2)

    k_batchelor_estimates_2 = np.nan * np.zeros_like(chi)
    cfunc_values_2: Float[ndarray, "time_slow k_test"] = np.zeros(
        (chi.shape[0], 30), dtype=float
    )
    for i, (mi, ma) in enumerate(zip(kmin_2, kmax_2)):
        k_test_2 = np.linspace(mi, ma, 30)  # k_batchelor values to test
        print(k_test_2)
        cfunc_values_2[i, :] = get_k_batchelor_costfunc(
            k_test_2,
            chi[i],
            k[i, :],
            Pk[i, :],
            Pnoise,
            viscosity_kinematic,
            diffusivity_temp,
        )
        idx = np.argmax(cfunc_values_2[i, :])
        k_batchelor_estimates_2[i] = k_test_2[idx]
    """
