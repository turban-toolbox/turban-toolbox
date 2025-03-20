import numpy as np
import scipy.signal



def calc_vsink(press,fs):
    """
    Calculates the sinking velocity by differentiating the pressure record

    Returns
    -------

    """

    vsink = gradient(press, 1 / fs)
    # TODO, interpolate linear missing values
    degree = 4
    cutoff_freq = .2 # Hz
    cutoff = cutoff_freq / (fs / 2) # non - dim
    # with Nyquist freq.
    [b, a] = scipy.signal.butter(degree, cutoff)
    vsink = scipy.signal.filtfilt(b,a,vsink)
    return vsink


def despike_std(x,win=1024,fac_std=3,max_spike_len=5):
    """
    Despikes x by linearly detrending x, calculating the standard deviation and removing everything that is
    thresh * standard deviation and shorter than max_spike_len.

    Parameters
    ----------
    x
    win
    thresh
    max_spike_len

    Returns
    -------

    """
    # Get the index of the spikes
    indspike = identify_spikes_std(x,win,fac_std,max_spike_len)
    t = np.arange(0,len(x))
    xint = np.interp(t,t[~indspike],x[~indspike])

    return xint

def identify_spikes_std(x,win=1024,fac_std=3,max_spike_len=5):
    """
    Identifies spikes by splitting x in windows of size win, detrendig and calculating the standard deviation of the
    detrended data in each window. If the differences between each datapoint and the mean exceeds fac_stc * std().

    Parameters
    ----------
    x
    win
    thresh

    Returns
    -------

    """
    ind_spike_all = np.zeros(np.shape(x),dtype=bool)
    for i in range(0,len(x), win):
        iend = i+win
        if(iend > len(x)):
            iend = len(x)
        xsnip    = x[i:iend] # Take a snippet of the time series
        xdetrend = scipy.signal.detrend(xsnip)
        xdiff    = xdetrend - np.mean(xdetrend)
        xthresh  = np.std(xdetrend) * fac_std
        indspike = abs(xdiff) >= xthresh
        # Lets split the indspike array into arrays that have a positive flank in the difference
        # np.diff([1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0])
        # array([ 0, -1,  1, -1,  0,  0,  1, -1,  1,  0,  0, -1,  1, -1])
        # [1, 1, 0, || 1, 0, 0, 0, || 1, 0, || 1, 1, 1, 0, || 1, 0]
        ind_diff = np.where(np.diff(indspike) == 1)[0] + 1
        indspike_split = np.split(indspike,ind_diff)
        # lets check which of the subarrays have more max_than spike_len values, because these would not be a spike anymore but a long signal
        for j, spike_section in enumerate(indspike_split):
            if(sum(spike_section) > max_spike_len):
                # Not a spike anymore, remove the indices
                if(j==0):
                    ind_tmp = ind_diff[j]
                    indspike[:ind_tmp] = False
                elif(j==(len(xdetrend)-1)):
                    ind_tmp = ind_diff[j]
                    indspike[j:] = False
                else:
                    ind_tmp0 = ind_diff[j-1]
                    ind_tmp1 = ind_diff[j]
                    indspike[ind_tmp0:ind_tmp1] = False

        ind_spike_all[i:iend] = indspike

    #ind_spike_all = ind_spike_all >= 1



    return ind_spike_all

def gradient(x,dt):
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
    #dxdt(i:len)=(-x(i:len)+ 4.*x(i-1:len-1) - 3.*x(i-2:len-2))./(2*dt)
    dxdt[2:] = (-x[2:] + 4 * x[1:-1] - 3 * x[0:-2])/( 2 * dt )
    dxdt[0] = (x[1] - x[0]) / dt
    dxdt[1] = (x[2] - x[0]) / ( 2 * dt )

    return dxdt
