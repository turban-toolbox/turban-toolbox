from abc import ABC, abstractmethod
import logging
from collections import defaultdict
import typing
from typing import Any

import numpy as np

from scipy.interpolate import PchipInterpolator as si_PchipInterpolator
import scipy.signal as ss

from turban.instruments.microrider import rsCommon as common
from turban.instruments.generic.channels import ChannelConfigBaseModel

logger = logging.getLogger(__name__)


class Converter(ABC):
    """Base class for detailed converter classes

    Parameters
    ----------
    config : ChannelConfigBaseModel
        validated channel configuration
    """

    def __init__(self, config: Any) -> None:
        # Any: each converter reads the fields of its own channel config class.
        self.config = config

    def __call__(
        self, raw: np.typing.NDArray[np.int16]
    ) -> np.typing.NDArray[np.float64]:
        return self.convert(self.decode(raw))

    def decode(self, raw: np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        """Interet the raw 16 bit word as counts"""
        if self.config.sign == "unsigned":
            return raw.view(np.uint16).astype(np.float64)
        else:
            return raw.astype(np.float64)

    @abstractmethod
    def convert(
        self, v: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        """Counts (float) -> physical units"""


class Piezo(Converter):
    """Specific converter method for Piezo type channels"""

    def convert(
        self, v: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        return v - self.config.a0


class Therm(Converter):
    """Specific converter method for thermistor type channels"""

    def convert(
        self, v: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        c = self.config
        Z = ((v - c.a) / c.b) * (c.adc_fs / 2**c.adc_bits) * 2 / (c.g * c.e_b)
        # Avoid taking log of negative numbers in case of broken sensor:
        Z = Z.clip(-0.6, 0.6, out=Z)
        R = (1 - Z) / (1 + Z)
        log_R = np.log(R)
        beta = c.beta if c.beta is not None else c.beta_1
        if beta is None:
            logger.error("No beta or beta_1 for this thermistor")
            raise ValueError("No beta or beta_1 for this thermistor")
        physical = 1 / c.t_0 + (1 / beta) * log_R
        if c.beta_2 is not None:
            physical += (1 / c.beta_2) * log_R**2
        if c.beta_3 is not None:
            physical += (1 / c.beta_3) * log_R**3
        physical = 1 / physical - 273.15
        return physical


class Shear(Converter):
    """Specific converter method for shear type channels"""

    def convert(
        self, v: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        c = self.config
        v_unit = (c.adc_fs / 2**c.adc_bits) * v + (c.adc_zero - c.sig_zero)
        v_unit /= c.diff_gain * c.sens * 2 * float(np.sqrt(2))
        return v_unit


class Poly(Converter):
    """Specific converter method for polynomial type channels"""

    def convert(
        self, v: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        polyVals: list[float] = []
        for i in range(9):
            p = getattr(self.config, f"coef{i:d}", None)
            if p is None:
                break
            polyVals.insert(0, p)
        return np.polyval(polyVals, v).astype(np.float64)


class Voltage(Converter):
    """Specific converter method for voltage type channels"""

    def convert(
        self, v: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        c = self.config
        return (c.adc_zero + v * c.adc_fs / 2**c.adc_bits) / c.g


class Incl(Converter):
    """Specific converter method channels to attitude type channels"""

    def decode(self, raw: np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        """
        The ADIS16209 inclinometer outputs words have the following format:

        bit15 = new data present
        bit14 = error flag
        bit[13:0] = X and Y inclination data, 2s-compliment (X and Y channels)
        bit[11:0] = temperature data, unsigned              (temperature channel)
        """
        # bits 0..12 have value information, and bit 13 the sign, When
        # we slect bits 0..12, and upgrade this to an int32, after
        # subtracting bit 13, it will work for negative values (X/Y),
        # but 12 bit positive values (T) will also fit.
        v_raw = np.bitwise_and(raw, (1 << 13) - 1).astype(np.int32) - np.bitwise_and(
            raw, (1 << 13)
        )
        return v_raw.astype(np.float64)

    def convert(
        self, v: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        return np.polyval([self.config.coef1, self.config.coef0], v).astype(np.float64)


class InclXY(Incl):
    """Specific converter method for attitude type channels"""


class InclT(Incl):
    """Specific converter method for attitude's temperature type channels"""


class Aem1g_a(Converter):
    """Specific converter method for Aem1g_a type channels"""

    def convert(
        self, v: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        c = self.config
        a = c.a / 100  # cm/s -> m/s
        b = c.b / 100  # cm/s -> m/s
        adc_zero = c.adc_fs / 2 if c.adc_zero is None else c.adc_zero
        v_unit: np.typing.NDArray[np.float64]
        v_unit = adc_zero + v * c.adc_fs / 2**c.adc_bits
        v_unit = a + b * v_unit
        v_unit -= c.bias
        if b < 1:
            m = """Provided coefficients (a,b) are not correct.  Each
instrument provides two sets of calibration results 
- one for analog output and one for digital output. 
The analog values should be used for this channel type."""
            logger.warning(m)
        return v_unit


class Aem1g_d(Converter):
    """Specific converter method for Aem1g_d type channels"""

    def decode(self, raw: np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        """The digital EM current word is always an unsigned 16 bit integer,
        irrespective of the configured sign."""
        return raw.view(np.uint16).astype(np.float64)

    def convert(
        self, v: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        c = self.config
        a = c.a / 100  # cm/s -> m/s
        b = c.b / 100  # cm/s -> m/s
        v_unit: np.typing.NDArray[np.float64]
        v_unit = a + b * v
        v_unit -= c.bias
        if b > 1:
            m = """Provided coefficients (a,b) are not correct.  Each
instrument provides two sets of calibration results 
- one for analog output and one for digital output. 
The digital values should be used for this channel type."""
            logger.warning(m)
        return v_unit


class PassThrough(Converter):
    """Pass through converter (also used for GND type channels)"""

    def convert(
        self, v: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        return v


class Deconvolve(object):
    r"""Description
      -----------
    Deconvolve a vector of pre-emphasized data (temperature, conductivity, or
    pressure) to yield high-resolution data. The pre-emphasized signal
    (x+gain*dx/dt) is low-pass filtered using appropriate initial conditions
    after the method described in Mudge and Lueck, 1994.

    For pressure, you must pass both 'X' and 'X_dX' to this function. Both
    vectors are needed to make a good estimate of the initial conditions for
    filtering. Initial conditions are very important for pressure because the
    gain is usually $\SI{\sim20}{\s}$, and errors caused by a poor choice of initial
    conditions can last for several hundred seconds! In addition, the
    high-resolution pressure is linearly adjusted to match the low-resolution
    pressure so that the factory calibration coefficients can later be used to
    convert this signal into physical units.

    The gains for all signal types are provided in the calibration report of
    your instrument.

      Examples

       >> C1_hres = deconvolve( 'C1_dC1', [], C1_dC1, fs_fast, setupfilestr )

    Deconvolve the micro-conductivity channel using the diff_gain parameter
    from the 'C1_dC1' channel section found within the supplied configuration
    string.  Because there is no channel without pre-emphasis, the argument is
    left empty.

       >> T1_hres = deconvolve( '', T1, T1_dT1, fs_fast, 1.034 )

    Deconvole the thermistor data using both channels with and without
    pre-emphasis.  Using both channels improves the initial deconvolution
    accuracy and compensates for slight variations in the calibration
    coefficients between the two channels. Note the explicitly provided value
    for diff_gain. One typically provides the configuration string and
    channel name as shown in the previous example.

    Mudge, T.D. and R.G. Lueck, 1994: Digital signal processing to enhance
    oceanographic observations, _J. Atmos. Oceanogr. Techn._, 11, 825-836.

    @image @images/pressure_deconvolution1 @Deconvolution example. @The green
    curve is from the normal pressure channel, and the blue curve is derived from
    the pre-emphasized pressure channel. This data is from a profiler that has impacted
    the soft bottom of a lake. Both signals are shown with full bandwidth (0 - 32 Hz)
    without any smoothing. The full-scale range of the pressure transducer is 500 dBar.

    @image @images/pressure_deconvolution2 @Deconvolution example two. @Same
    as previous figure but with zoom-in on only the high-resolution pressure. Again,
    full bandwidth without any smoothing.

    @image @images/pressure_deconvolution3 @The rate of change of pressure derived
    from the normal pressure signal and the high-resolution pressure signals using
    the gradient function. @Full bandwidth signals of the rate of change of
    pressure. The normal pressure signal (green) produces a fall-rate that is
    useless without extensive smoothing because it is not even
    monotonic. The rate of change of the high-resolution pressure (blue) is smooth, always
    positive, and, therefore, the high-resolution pressure itself, can be used directly for
    plotting other variables as a function pressure (or depth). The
    high-resolution rate of change of pressure has been multiplied by 10 for visual
    clarity. The fall-rate is about $\SI{0.17}{\m\per\s}$."""

    def __init__(
        self,
        X_dX: np.typing.NDArray[np.float64],
        X: np.typing.NDArray[np.float64],
        fs: float,
        diff_gain: float,
    ):
        self.fs = fs
        self.diff_gain = diff_gain
        X = self.interpolate(X_dX, X)
        self.X_hires = self.bw_filter(X_dX, X)
        self.X = X

    def bw_filter(
        self, X_dX: np.typing.NDArray[np.float64], X: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        fc = 1 / (2 * np.pi * self.diff_gain)
        b, a = ss.butter(1, fc / (self.fs / 2))
        if not X is None:
            p = np.polyfit(X, X_dX, 1)
            if p[0] < -0.5:
                X_dX = -X_dX
            z = ss.lfiltic(b, a, X[:1], X_dX[:1])
        else:
            tm = np.arange(0, 2 * self.diff_gain, 1 / self.fs)
            p = np.polyfit(tm, X_dX[: tm.shape[0]], 1)
            previousOutput = np.array([p[1] - self.diff_gain * p[0]])
            z = ss.lfiltic(b, a, previousOutput, X_dX[:1])
        X_hires, zf = ss.lfilter(b, a, X_dX, zi=z)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %%%%% For the pressure, regress against the low-resolution vector to %%%%%
        # %%%%% remove the small offset in the derivative circuit.             %%%%%
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        #  Algorithm for signals that have both with and without pre-emphasis data.
        #  1) Interpolate the non pre-emphasis signal so we can perform a polyfit.
        #  2) Deconvolve using the first "X" data point for the initial condition.
        #  3) Polyfit the deconvolved signal to the non pre-emphasis signal.
        #  4) Calculate new initial conditions after applying an offset to X based
        #     from the polyfit applied in step 3.
        #  5) Perform a new deconvolve with updated initial conditions.
        #  6) Apply the polynomial from step 3 to the new data.  A new polyfit is
        #     not required - they will be almost identicle.

        if not X is None:
            p = np.polyfit(X_hires, X, 1)
            p2 = np.array([2 - p[0], -p[1]])
            initialOutput = np.polyval(p2, X[:1])
            z = ss.lfiltic(b, a, initialOutput, X_dX[:1])
            X_hires, zf = ss.lfilter(b, a, X_dX, zi=z)
            X_hires = np.polyval(p, X_hires).astype(np.float64)
        return X_hires

    def interpolate(
        self, X_dX: np.typing.NDArray[np.float64], X: np.typing.NDArray[np.float64]
    ) -> typing.Any:
        # only interpolate if X and X_dX are not of equal length
        if X is None or (X.shape == X_dX.shape):
            return X  # nothing to do
        f_slow = self.fs * X.shape[0] / X_dX.shape[0]
        t_slow = np.arange(X.shape[0]) / f_slow
        t_fast = np.arange(X_dX.shape[0]) / self.fs
        ifun = si_PchipInterpolator(t_slow, X, extrapolate=True)
        return ifun(t_fast)


def get_converter(channel_config: ChannelConfigBaseModel) -> type[Converter]:
    """Converter factory

    Parameters
    ----------
    channel_config : ChannelConfig
        channel configuration dataclass

    Returns
    -------
    Converter object, relevant for the requested channel type.

    Note : this code uses the match/case structure and requires therefore python 3.10+
    """
    converter: type[Converter]
    try:
        sensor_type = channel_config.type
    except KeyError:
        sensor_type = "none"
    match sensor_type:
        case "piezo":
            converter = Piezo
        case "gnd":
            converter = PassThrough
        case "therm":
            converter = Therm
        case "shear":
            converter = Shear
        case "poly":
            converter = Poly
        case "voltage":
            converter = Voltage
        case "inclxy":
            converter = InclXY
        case "inclt":
            converter = InclT
        case "aem1g_a":
            converter = Aem1g_a
        case "aem1g_d":
            converter = Aem1g_d
        case "none":
            converter = PassThrough
        case _:
            converter = PassThrough

    return converter
