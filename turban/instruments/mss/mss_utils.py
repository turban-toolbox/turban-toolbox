import numpy as np
import scipy.signal
import numpy as np
from jaxtyping import Num
from numpy import ndarray, newaxis
from scipy.signal import butter, lfilter, lfiltic


def calc_shear(shear, vsink, density, fs):
    """Calculate physical shear from sensor data.

    High-pass filters at 1 Hz, computes time gradient, then applies
    the relationship: shear = dshear/dt / (density * vsink^2).

    Parameters
    ----------
    shear : array_like
        Raw shear probe signal.
    vsink : array_like
        Sinking velocity in m/s.
    density : array_like
        Water density in kg/m³.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    ndarray
        Physical shear.
    """
    dt = 1 / fs
    degree = 4
    cutoff_Fs = 1  # 1 Hz
    cutoff = cutoff_Fs / (fs / 2)  # non - dim with Nyquist freq.
    [b, a] = scipy.signal.butter(degree, cutoff, "high")

    shear_hp = scipy.signal.filtfilt(b, a, shear)
    # calculate time gradient of raw shear
    dshdt = np.gradient(shear_hp, dt)
    # screen for spikes
    # dshdt_desp = despike_std(dshdt, 1024, 4)
    dshdt_desp = dshdt  # do not despike at level1!

    vsink_tmp = vsink.copy()
    vsink_tmp[vsink_tmp == 0] = np.nan

    shear = dshdt_desp * (density ** (-1)) * (vsink_tmp ** (-2))
    return shear


def calc_vsink(press, fs, f_low=0.2):
    """Calculate sinking velocity by low-pass filtering and differentiating pressure.

    Parameters
    ----------
    press : array_like
        Pressure time series.
    fs : float
        Sampling frequency in Hz.
    f_low : float, optional
        Low-pass cutoff frequency in Hz. Default is 0.2.

    Returns
    -------
    vsink : ndarray
        Sinking velocity estimated as the time derivative of low-pass filtered pressure.
    """
    dt = 1 / fs
    # Low pass the pressure signal
    degree = 4
    cutoff_freq = f_low  # Hz
    cutoff = cutoff_freq / (fs / 2)  # non - dim
    # with Nyquist freq.
    [b, a] = scipy.signal.butter(degree, cutoff)
    press_low = scipy.signal.filtfilt(b, a, press)
    vsink = np.gradient(press_low, dt)
    return vsink


def calc_vsink_legacy(press, fs):
    """Calculate sinking velocity by differentiating pressure record.

    Uses the legacy gradient function with low-pass filtering at 0.2 Hz.

    Parameters
    ----------
    press : array_like
        Pressure time series.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    ndarray
        Sinking velocity.
    """

    vsink = gradient(press, 1 / fs)
    # TODO, interpolate linear missing values
    degree = 4
    cutoff_freq = 0.2  # Hz
    cutoff = cutoff_freq / (fs / 2)  # non - dim
    # with Nyquist freq.
    [b, a] = scipy.signal.butter(degree, cutoff)
    vsink = scipy.signal.filtfilt(b, a, vsink)
    return vsink


def despike_std(x, win=1024, fac_std=3, max_spike_len=5):
    """Remove spikes from a time series using standard deviation thresholding.

    Detrends the input in windows, identifies points exceeding fac_std times the
    standard deviation within each window, and removes spikes shorter than
    max_spike_len by interpolation.

    Parameters
    ----------
    x : array_like
        Time series to despike.
    win : int, optional
        Window size in samples. Default is 1024.
    fac_std : float, optional
        Standard deviation factor threshold. Default is 3.
    max_spike_len : int, optional
        Maximum spike length in samples. Spikes longer than this are retained.
        Default is 5.

    Returns
    -------
    ndarray
        Despiked time series with spikes removed by linear interpolation.
    """
    # Get the index of the spikes
    indspike = identify_spikes_std(x, win, fac_std, max_spike_len)
    t = np.arange(0, len(x))
    xint = np.interp(t, t[~indspike], x[~indspike])

    return xint


def identify_spikes_std(x, win=1024, fac_std=3, max_spike_len=5):
    """Identify spikes in a time series using windowed standard deviation.

    Splits the time series into windows, detrends each window, and flags points
    that deviate from the window mean by more than fac_std times the window
    standard deviation. Only spikes shorter than max_spike_len samples are marked.

    Parameters
    ----------
    x : array_like
        Time series to analyze.
    win : int, optional
        Window size in samples. Default is 1024.
    fac_std : float, optional
        Standard deviation factor threshold. Default is 3.
    max_spike_len : int, optional
        Maximum spike length in samples. Spikes longer than this are not flagged.
        Default is 5.

    Returns
    -------
    ndarray of bool
        Boolean mask where True indicates a spike.
    """
    ind_spike_all = np.zeros(np.shape(x), dtype=bool)
    for i in range(0, len(x), win):
        iend = i + win
        if iend > len(x):
            iend = len(x)
        xsnip = x[i:iend]  # Take a snippet of the time series
        xdetrend = scipy.signal.detrend(xsnip)
        xdiff = xdetrend - np.mean(xdetrend)
        xthresh = np.std(xdetrend) * fac_std
        indspike = abs(xdiff) >= xthresh
        # Lets split the indspike array into arrays that have a positive flank in the difference
        # np.diff([1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0])
        # array([ 0, -1,  1, -1,  0,  0,  1, -1,  1,  0,  0, -1,  1, -1])
        # [1, 1, 0, || 1, 0, 0, 0, || 1, 0, || 1, 1, 1, 0, || 1, 0]
        ind_diff = np.where(np.diff(indspike) == 1)[0] + 1
        indspike_split = np.split(indspike, ind_diff)
        # lets check which of the subarrays have more max_than spike_len values, because these would not be a spike anymore but a long signal
        for j, spike_section in enumerate(indspike_split):
            if sum(spike_section) > max_spike_len:
                # Not a spike anymore, remove the indices
                if j == 0:
                    ind_tmp = ind_diff[j]
                    indspike[:ind_tmp] = False
                elif j == (len(xdetrend) - 1):
                    ind_tmp = ind_diff[j]
                    indspike[j:] = False
                else:
                    ind_tmp0 = ind_diff[j - 1]
                    ind_tmp1 = ind_diff[j]
                    indspike[ind_tmp0:ind_tmp1] = False

        ind_spike_all[i:iend] = indspike

    # ind_spike_all = ind_spike_all >= 1

    return ind_spike_all


def gradient_legacy(x, dt):
    """
    Calculates a a finite difference gradient with an order of dt**2
    see also https://en.wikipedia.org/wiki/Finite_difference#Relation_with_derivatives

    Parameters
    ----------
    x
    dt

    Returns
    -------
    dxdt: numpy array of the finite difference
    """

    dxdt = np.zeros((len(x)))
    print(np.asarray(x))
    print(np.shape(x))
    print(np.shape(dxdt))
    print(np.shape(x[2:]), np.shape(x[1:-1]), np.shape(x[0:-2]))
    print(np.shape(dxdt[2:]))
    print(np.shape((-x[2:] + 4 * x[1:-1] - 3 * x[0:-2]) / (2 * dt)))
    print(np.shape((-x[2:] + 4 * x[1:-1] - 3 * x[0:-2])))
    # dxdt(i:len)=(-x(i:len)+ 4.*x(i-1:len-1) - 3.*x(i-2:len-2))./(2*dt)
    dxdt[2:] = (-x[2:] + 4 * x[1:-1] - 3 * x[0:-2]) / (2 * dt)
    dxdt[0] = (x[1] - x[0]) / dt
    dxdt[1] = (x[2] - x[0]) / (2 * dt)

    return dxdt


def deconvolve_mss_ntchp(
    x: Num[ndarray, "time"],
    x_emph: Num[ndarray, "time"],
    sampfreq: float,
    gain: float = 1.5,
):
    """Deconvolve the MSS NTC high-pass pre-emphasis filter.

    Parameters
    ----------
    x : ndarray, shape (time,)
        Original (unemphasised) signal, used to set initial filter conditions.
    x_emph : Num[ndarray, "time"]
        Pre-emphasised signal to be deconvolved.
    sampfreq : float
        Sampling frequency in Hz.
    gain : float, optional
        Time constant of the high-pass pre-emphasis filter in seconds.
        Default is 1.5.

    Returns
    -------
    ndarray, shape (time,)
        Deconvolved signal with the emphasis removed.
    """
    cutoff_freq_Hz = 1 / (2 * np.pi * gain)
    # cutoff_freq_Hz = 0.5
    cutoff_nondim = cutoff_freq_Hz / (sampfreq / 2)
    b, a = butter(N=1, Wn=cutoff_nondim, btype="low")
    zi = lfiltic(b, a, x[:1], x_emph[:1])  # initial conditions
    x_d, _ = lfilter(b, a, x_emph, zi=zi)  # deconvoluted
    return x_d
