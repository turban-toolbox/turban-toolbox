"""Channel configurations and conversions to physical units for the MicroRider.

Each channel type (the ``type`` key of a ``[channel]`` section in the setup
string) is represented by one pydantic model that holds the calibration
parameters, their defaults, and the ``convert`` method that turns raw counts
into physical units. ``make_channel_config`` selects the model for a section.
"""

import re
from typing import Annotated, Any, ClassVar, Literal, Self
import typing

import numpy as np

from pydantic import BeforeValidator, ConfigDict, ValidationError, model_validator
from scipy.interpolate import PchipInterpolator as si_PchipInterpolator
import scipy.signal as ss

from turban.instruments.generic.channels import ChannelConfigBaseModel, RawArray
from turban.utils.logging import get_logger

logger = get_logger(__name__)


def _number_to_str(v: Any) -> Any:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return v


# Serial numbers like "487" are parsed as numbers by the setup string parser.
# Note: formatting is lost by then, e.g. "059" arrives as 59 and becomes "59".
SerialNumber = Annotated[str, BeforeValidator(_number_to_str)]


def _counts_to_volts(
    v: RawArray, adc_fs: float, adc_bits: int
) -> np.typing.NDArray[np.float64]:
    return v.astype(np.float64) * adc_fs / 2**adc_bits


class MicroRiderChannelConfig(ChannelConfigBaseModel):
    """Base class for MicroRider channel configurations

    Unknown fields are rejected. Keys in the setup string that a channel type
    does not use are dropped by ``make_channel_config`` before validation;
    keys in user-supplied overrides are not, so a misspelled override fails.

    Without a more specific type, no conversion is applied.
    """

    model_config = ConfigDict(extra="forbid")


class PreEmphasisChannel(MicroRiderChannelConfig):
    """Pre-emphasised channel (T1_dT1, P_dP, ...)

    Not converted itself; only provides ``diff_gain`` for the deconvolution.
    The deconvolved channel is converted with the config of the channel
    without pre-emphasis (T1, P, ...).
    """

    diff_gain: float

    # Names of pre-emphasised channels that are deconvolved: T1_dT1, T2_dT2, P_dP
    name_regex: ClassVar[re.Pattern[str]] = re.compile(r"([TP]\w?)_d\1")


class PassThroughChannel(MicroRiderChannelConfig):
    """Fallback for unknown channel types or incomplete calibrations"""


class GndChannel(MicroRiderChannelConfig):
    """Ground channel, no conversion"""

    type: Literal["gnd"]
    units: str = "counts"


class PiezoChannel(MicroRiderChannelConfig):
    """Piezo-accelerometer channel (Ax, Ay)"""

    type: Literal["piezo"]
    a0: float = 0.0
    units: str = "counts"

    def convert(self, v: RawArray) -> np.typing.NDArray[np.float64]:
        return v.astype(np.float64) - self.a0


class ShearChannel(MicroRiderChannelConfig):
    """Shear probe channel (sh1, sh2)"""

    type: Literal["shear"]
    adc_fs: float
    adc_bits: int
    diff_gain: float
    sens: float
    adc_zero: float = 0.0
    sig_zero: float = 0.0
    sn: SerialNumber = ""
    cal_date: str = ""
    units: str = "s-1"

    def convert(self, v: RawArray) -> np.typing.NDArray[np.float64]:
        volts = _counts_to_volts(v, self.adc_fs, self.adc_bits)
        volts += self.adc_zero - self.sig_zero
        return volts / (self.diff_gain * self.sens * 2 * float(np.sqrt(2)))


class ThermChannel(MicroRiderChannelConfig):
    """Thermistor channel (T1, T2)"""

    type: Literal["therm"]
    adc_fs: float
    adc_bits: int
    a: float
    b: float
    g: float
    e_b: float
    t_0: float
    beta: float | None = None
    beta_1: float | None = None
    beta_2: float | None = None
    beta_3: float | None = None
    sn: SerialNumber = ""
    cal_date: str = ""
    units: str = "degC"

    @model_validator(mode="after")
    def _require_beta(self) -> Self:
        if self.beta is None and self.beta_1 is None:
            raise ValueError(f"{self.name}: thermistor needs beta or beta_1")
        return self

    def convert(self, v: RawArray) -> np.typing.NDArray[np.float64]:
        z = (v.astype(np.float64) - self.a) / self.b
        z *= (self.adc_fs / 2**self.adc_bits) * 2 / (self.g * self.e_b)
        # Avoid taking log of negative numbers in case of broken sensor:
        z = z.clip(-0.6, 0.6)
        log_r = np.log((1 - z) / (1 + z))
        beta = self.beta if self.beta is not None else self.beta_1
        assert beta is not None  # guaranteed by _require_beta
        inv_t = 1 / self.t_0 + log_r / beta
        if self.beta_2 is not None:
            inv_t += log_r**2 / self.beta_2
        if self.beta_3 is not None:
            inv_t += log_r**3 / self.beta_3
        return 1 / inv_t - 273.15


class PolyChannel(MicroRiderChannelConfig):
    """Polynomial channel (P, PV, ...): coef0 + coef1*v + coef2*v**2 + ..."""

    type: Literal["poly"]
    coef0: float
    coef1: float | None = None
    coef2: float | None = None
    coef3: float | None = None
    coef4: float | None = None
    coef5: float | None = None
    coef6: float | None = None
    coef7: float | None = None
    coef8: float | None = None
    cal_date: str = ""

    @property
    def coefs(self) -> list[float]:
        """Coefficients coef0, coef1, ... up to the first one not set"""
        coefs = []
        for i in range(9):
            c = getattr(self, f"coef{i}")
            if c is None:
                break
            coefs.append(c)
        return coefs

    def convert(self, v: RawArray) -> np.typing.NDArray[np.float64]:
        return np.polynomial.polynomial.polyval(v.astype(np.float64), self.coefs)


class VoltageChannel(MicroRiderChannelConfig):
    """Voltage channel (V_Bat, EMC_Cur)"""

    type: Literal["voltage"]
    adc_fs: float
    adc_bits: int
    adc_zero: float = 0.0
    g: float = 1.0
    units: str = "V"

    def convert(self, v: RawArray) -> np.typing.NDArray[np.float64]:
        volts = _counts_to_volts(v, self.adc_fs, self.adc_bits)
        return (self.adc_zero + volts) / self.g


class _InclChannel(MicroRiderChannelConfig):
    """ADIS16209 inclinometer channel"""

    coef0: float
    coef1: float

    def convert(self, v: RawArray) -> np.typing.NDArray[np.float64]:
        if v.dtype == ">i2":
            v = v.astype(np.int16)
        if not v.dtype == np.int16:
            raise ValueError("Unexpected datatype encountered.")
        # old data and errors are all ignored. Should we set all those elements to nan?
        v_raw = Adis().convert(v.astype(np.int16))
        return self.coef0 + self.coef1 * v_raw


class InclXYChannel(_InclChannel):
    """Inclinometer angle channel (Incl_X, Incl_Y)"""

    type: Literal["inclxy"]
    units: str = "deg"


class InclTChannel(_InclChannel):
    """Inclinometer temperature channel (Incl_T)"""

    type: Literal["inclt"]
    coef0: float = 624.0
    coef1: float = -0.47
    units: str = "degC"


class _Aem1gChannel(MicroRiderChannelConfig):
    """JFE Advantech AEM1-G electromagnetic current meter"""

    a: float  # cm/s, as given in the calibration report
    b: float  # cm/s per V (analog) or per count (digital)
    bias: float = 0.0  # m/s
    sn: SerialNumber = ""
    cal_date: str = ""
    units: str = "m s-1"


class Aem1gAnalogChannel(_Aem1gChannel):
    type: Literal["aem1g_a"]
    adc_fs: float
    adc_bits: int
    adc_zero: float | None = None  # defaults to adc_fs / 2

    @model_validator(mode="after")
    def _check_coefs(self) -> Self:
        if self.b / 100 < 1:
            logger.warning(
                f"{self.name}: provided coefficients (a, b) are not correct. Each "
                "instrument provides two sets of calibration results - one for "
                "analog output and one for digital output. The analog values "
                "should be used for this channel type."
            )
        return self

    def convert(self, v: RawArray) -> np.typing.NDArray[np.float64]:
        adc_zero = self.adc_fs / 2 if self.adc_zero is None else self.adc_zero
        volts = adc_zero + _counts_to_volts(v, self.adc_fs, self.adc_bits)
        return self.a / 100 + self.b / 100 * volts - self.bias


class Aem1gDigitalChannel(_Aem1gChannel):
    type: Literal["aem1g_d"]

    @model_validator(mode="after")
    def _check_coefs(self) -> Self:
        if self.b / 100 > 1:
            logger.warning(
                f"{self.name}: provided coefficients (a, b) are not correct. Each "
                "instrument provides two sets of calibration results - one for "
                "analog output and one for digital output. The digital values "
                "should be used for this channel type."
            )
        return self

    def convert(self, v: RawArray) -> np.typing.NDArray[np.float64]:
        u = v.astype(">u2").astype(np.float64)  # unsigned 16 bit integer
        return self.a / 100 + self.b / 100 * u - self.bias


# Channel type (the ``type`` key in the setup string) -> configuration class
CHANNEL_TYPES: dict[str, type[MicroRiderChannelConfig]] = {
    "gnd": GndChannel,
    "piezo": PiezoChannel,
    "shear": ShearChannel,
    "therm": ThermChannel,
    "poly": PolyChannel,
    "voltage": VoltageChannel,
    "inclxy": InclXYChannel,
    "inclt": InclTChannel,
    "aem1g_a": Aem1gAnalogChannel,
    "aem1g_d": Aem1gDigitalChannel,
}


def _known_fields(
    cls: type[MicroRiderChannelConfig], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Drops setup string keys that the configuration class does not use"""
    unused = set(kwargs) - set(cls.model_fields)
    if unused:
        logger.debug(f"{kwargs.get('name')!r}: ignoring setup keys {sorted(unused)}")
    return {k: v for k, v in kwargs.items() if k in cls.model_fields}


def make_channel_config(
    section: dict[str, Any], override: dict[str, Any] | None = None
) -> MicroRiderChannelConfig:
    """Builds a channel configuration from a ``[channel]`` section

    Parameters
    ----------
    section : dict[str, Any]
        key-value pairs of one ``[channel]`` section of the setup string.
    override : dict[str, Any] | None
        user-supplied values that replace those of the setup string,
        e.g. ``{"name": "sh1", "sens": 0.0712}``.

    Returns
    -------
    MicroRiderChannelConfig
        configuration of the class matching the channel type. Pre-emphasised
        channels (T1_dT1, P_dP, ...) get a PreEmphasisChannel, selected by name
        as they share the type of the channel without pre-emphasis. Channels of
        unknown type, or with an incomplete calibration in the setup string, get
        a PassThroughChannel, which leaves the data in counts.

    Raises
    ------
    pydantic.ValidationError
        if a pre-emphasised channel has no diff_gain, or if the override has a
        field that the channel type does not have, or an invalid value.
    """
    kwargs = {k.lower(): v for k, v in section.items()}
    name = str(kwargs.get("name", ""))
    cls: type[MicroRiderChannelConfig]
    if PreEmphasisChannel.name_regex.fullmatch(name):
        cls = PreEmphasisChannel
    else:
        cls = CHANNEL_TYPES.get(str(kwargs.get("type", "")), PassThroughChannel)
    kwargs = _known_fields(cls, kwargs)
    if override is not None:
        # not filtered: unknown fields in user input must fail.
        kwargs.update({k.lower(): v for k, v in override.items()})
        return cls.model_validate(kwargs)
    if cls is PreEmphasisChannel:  # without diff_gain, deconvolution is impossible
        return cls.model_validate(kwargs)
    try:
        return cls.model_validate(kwargs)
    except ValidationError as e:
        logger.warning(
            f"Channel {name!r} is not converted into units "
            f"(incomplete configuration): {e}"
        )
        return PassThroughChannel.model_validate(
            _known_fields(PassThroughChannel, kwargs)
        )


class Adis(object):
    """
    The ADIS16209 inclinometer outputs words that combine data with status flags.
    The data and status flags must be seperated before the data can be converted
    into physical units.  This function extracts the data and converts it into
    valid 16bit, 2s-compliment words.

    ADIS16209 words have the following format:

        bit15 = new data present
        bit14 = error flag
    bit[13:0] = X and Y inclination data, 2s-compliment (X and Y channels)
    bit[11:0] = temperature data, unsigned              (temperature channel)
    """

    def __init__(self) -> None:
        # self.errorFlag : np.typing.NDArray[np.int16] = np.array([], dtype=np.int16)
        # self.oldFlag : np.typing.NDArray[np.int16] = np.array([], dtype=np.int16)
        pass

    def convert(self, v: np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        u = v.astype(">u2")  # unsigned 16 bit integer
        b = np.bitwise_and(u, 1 << 15) // (1 << 15)  # b==1: new data b==0: old data
        # self.oldFlag = np.where(b==0)[0]
        b = np.bitwise_and(u, 1 << 14) // (1 << 14)  # b==1: error b==0: no error
        # self.errorFlag = np.where(b==1)[0]
        b = np.bitwise_and(u, (1 << 13) + (1 << 12))
        v_raw: np.typing.NDArray[np.int16] | np.typing.NDArray[np.uint16]
        if np.all(b == 0):
            # dtype = '>u2' # big endian unsigned int 2 byte (16 bit)
            v_raw = np.bitwise_and(u, (1 << 13) - 1)
            v_raw -= np.bitwise_and(u, 1 << 13)  # two's compliment.
            v_raw = v_raw.astype(np.uint16)
        else:
            # dtype = '>i2' # big endian signed int 2 byte (16 bit)
            v_raw = np.bitwise_and(u, (1 << 13) - 1)
            v_raw -= np.bitwise_and(u, 1 << 13)  # two's compliment.
            v_raw = v_raw.astype(np.int16)
        return v_raw.astype(np.float64)


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
