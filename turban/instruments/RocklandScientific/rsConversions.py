from abc import ABC, abstractmethod
import logging
from collections import defaultdict

import numpy as np

from scipy.interpolate import PchipInterpolator as si_PchipInterpolator
import scipy.signal as ss

from . import rsCommon as common

logger = logging.getLogger(__name__)


class Converter(ABC):
    '''Base class for detailed converter classes
    
    Parameters
    ----------
    config : ChannelConfig
        channel configuration dataclass
    '''
    def __init__(self, config: common.ChannelConfigABC) -> None:
        self.config: common.ChannelConfigABC = config
        self.defaults: common.ChannelConfigABC
    
    def __call__(self, v: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        return self.convert(v)

    def get_parameter(self, p: str) -> int | float:
        '''Looks up a parameter setting from the configuration

        Parameters
        ----------
        p : string
            parameter name
        
        Returns
        float 
            value of parameter setting.

        The method returs the value set in the configuration, or a default value when no 
        value is configured. If also no default value is given, a ValueError is raised.
        '''
        # return the value if it is configured
        if self.config.is_set(p):
            return_value = getattr(self.config, p)
            if isinstance(return_value, int) or isinstance(return_value, float):
                return return_value
            else:
                raise ValueError("Unexpected config value encountered.")
        # not configured, return the value from the defaults, if configured
        if self.defaults.is_set(p):
            return_value = getattr(self.defaults, p)
            if isinstance(return_value, int) or isinstance(return_value, float):
                return return_value
            else:
                raise ValueError("Unexpected default value encountered.")
        else:
            raise ValueError(f"Parameter {p} is required, but not given in config or default dict")

    def get_optional_parameter(self, p: str) -> int | float | None:
        '''Looks up a parameter setting from the configuration

        Parameters
        ----------
        p : string
            parameter name
        
        Returns
        float 
            value of parameter setting.

        The method returs the value set in the configuration, or a default value when no 
        value is configured. If also no default value is given None is returned.
        '''
        try:
            return_value = self.get_parameter(p)
        except ValueError:
            return_value = None
        return return_value
        

    @abstractmethod
    def convert(self, v: np.typing.NDArray[np.float64] | np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        pass





class Piezo(Converter):
    '''Specific converter method for Piezo type channels'''
    def __init__(self, config: common.ChannelConfigABC):
        super().__init__(config)
        self.defaults = common.ChannelConfigPiezo(a0=0., units='[ counts ]')
        
    def convert(self, v: np.typing.NDArray[np.float64] | np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        a0 = self.get_parameter('a0')
        v_unit = v.astype(np.float64) - a0
        return v_unit

class Gnd(Converter):
    '''Specific converter method for GND type channels
    
    Does not do any conversion.
    '''
    def __init__(self, config: common.ChannelConfigABC):
        super().__init__(config)
        self.defaults = common.ChannelConfig(units='[ counts ]')

    def convert(self, v:np.typing.NDArray[np.float64] | np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        return v.astype(np.float64)

class Therm(Converter):
    def __init__(self, config: common.ChannelConfigABC):
        '''Specific converter method for thermistor type channels'''
            
        super().__init__(config)
        self.defaults = common.ChannelConfigThermistor(units='[ °C ]')

    def convert(self, v:np.typing.NDArray[np.float64] | np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        a = self.get_parameter('a')
        b = self.get_parameter('b')
        adc_fs = self.get_parameter('adc_fs')
        adc_bits = self.get_parameter('adc_bits')
        g = self.get_parameter('g')
        e_b = self.get_parameter('e_b')
        t_0 = self.get_parameter('t_0')

        Z = ((v - a)/b) * (adc_fs/2**adc_bits) * 2 / (g * e_b)
        # Avoid taking log of negative numbers in case of broken sensor:
        Z = Z.clip(-0.6, 0.6, out=Z)
        R = (1 - Z)/(1 + Z)
        log_R = np.log(R)
        beta = self.get_optional_parameter('beta') or self.get_optional_parameter('beta_1')
        if beta is None:
            logger.error('No beta or beta_1 for this thermistor')
            ValueError('No beta or beta_1 for this thermistor')
        else:
            physical = 1/t_0 + (1/beta) * log_R

        beta_2 = self.get_optional_parameter('beta_2')
        if not beta_2 is None:
            physical += (1/beta_2) * log_R**2
            
        beta_3 = self.get_optional_parameter('beta_3')
        if not beta_3 is None:
            physical += (1/beta_3) * log_R**3

        physical = 1/physical - 273.15
        return physical

class Shear(Converter):
    '''Specific converter method for shear type channels'''
    
    def __init__(self, config: common.ChannelConfigABC):
        super().__init__(config)
        self.defaults = common.ChannelConfigShear(adc_zero=0., sig_zero=0., units='[ s^{-1} ]')
                
    def convert(self, v:np.typing.NDArray[np.float64] | np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        v = v.astype(np.float64)
        adc_zero = self.get_parameter('adc_zero')
        sig_zero = self.get_parameter('sig_zero')
        adc_fs = self.get_parameter('adc_fs')
        adc_bits = self.get_parameter('adc_bits')
        diff_gain = self.get_parameter('diff_gain')
        sens = self.get_parameter('sens')
        v_unit = (adc_fs / 2**adc_bits) * v + (adc_zero - sig_zero)
        v_unit /= diff_gain * sens * 2 * float(np.sqrt(2))
        return v_unit

    
class Poly(Converter):
    '''Specific converter method for polynomial type channels'''
    
    def __init__(self, config: common.ChannelConfigABC):
        super().__init__(config)
        self.defaults = common.ChannelConfigPressure(units=' ')
        
    def convert(self, v:np.typing.NDArray[np.float64] | np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        polyVals : list[float | int] = []
        for i in range(9):
            p = self.get_optional_parameter(f"coef{i:d}")
            if p is None:
                break
            polyVals.insert(0, p)
        v_unit = np.polyval(polyVals, v).astype(np.float64)
        return v_unit

class Voltage(Converter):
    '''Specific converter method for voltage type channels'''
    def __init__(self, config: common.ChannelConfigABC):
        super().__init__(config)
        self.defaults = common.ChannelConfigVoltage(adc_zero=0., g=1.0, units = '[ V ]')
        
    def convert(self, v:np.typing.NDArray[np.float64] | np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        adc_zero = self.get_parameter("adc_zero")
        gain = self.get_parameter("g")
        adc_fs = self.get_parameter("adc_fs")
        adc_bits = self.get_parameter("adc_bits")
        v_unit = (adc_zero + v * adc_fs/2**adc_bits) / gain
        return v_unit

class Incl(Converter):
    '''Specific converter method common to attitude type channels'''

    def __init__(self, config: common.ChannelConfigABC):
        super().__init__(config)
        self.adis = Adis()

    def convert(self, v:np.typing.NDArray[np.float64] | np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        coefs = [self.get_parameter('coef1'),
                 self.get_parameter('coef0')]
        if v.dtype == '>i2':
            v = v.astype(np.int16)
        if  not v.dtype == np.int16:
            raise ValueError("Unexpected datatype encountered.")
        v_raw = self.adis.convert(v.astype(np.int16))
        # old data and errors are all ignored. Should we set all those elements to nan?
        v_unit = np.polyval(coefs, v_raw).astype(np.float64)
        return v_unit
    
class InclXY(Incl):
    '''Specific converter method for attitude type channels'''

    def __init__(self, config: common.ChannelConfigABC):
        super().__init__(config)
        self.defaults = common.ChannelConfigInclinometer(units = '[ ° ]')
        
       

class InclT(Incl):
    '''Specific converter method for attitude's temperature type channels'''

    def __init__(self, config: common.ChannelConfigABC):
        super().__init__(config)
        self.defaults = common.ChannelConfigInclinometer(coef0=624, coef1=-0.47, units = '[ °C ]')


class Aem1g_a(Converter):
    '''Specific converter method for Aem1g_a type channels'''

    def __init__(self, config: common.ChannelConfigABC):
        super().__init__(config)
        self.defaults = common.ChannelConfigU_EM(bias=0, units='[ m s^{-1} ]')
        
    def convert(self, v: np.typing.NDArray[np.int16] | np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        bias = self.get_parameter('bias')
        adc_fs  = self.get_parameter('adc_fs')
        adc_bits  = self.get_parameter('adc_bits')
        adc_zero = self.get_optional_parameter('adc_zero')
        a = self.get_parameter('a') / 100 # cm/s -> m/s
        b = self.get_parameter('b') / 100 # cm/s -> m/s
        v_unit : np.typing.NDArray[np.float64]
        if adc_zero is None: adc_zero = adc_fs / 2
        v_unit = adc_zero + v * (adc_fs / 2**adc_bits)
        v_unit = a + b * v_unit
        v_unit -= bias
        if b<1:
            m = '''Provided coefficients (a,b) are not correct.  Each
instrument provides two sets of calibration results 
- one for analog output and one for digital output. 
The analog values should be used for this channel type.'''
            logger.warning(m)
        return v_unit


    
class PassThrough(Converter):
    ''' Pass through converter '''
    def __init__(self, config: common.ChannelConfigABC):
        super().__init__(config)
        self.defaults = common.ChannelConfig()
        
    def convert(self, v:np.typing.NDArray[np.int16] | np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        return v.astype(np.float64)
            
    
class Adis(object):
    '''
    The ADIS16209 inclinometer outputs words that combine data with status flags.
    The data and status flags must be seperated before the data can be converted
    into physical units.  This function extracts the data and converts it into
    valid 16bit, 2s-compliment words.

    ADIS16209 words have the following format:

        bit15 = new data present
        bit14 = error flag
    bit[13:0] = X and Y inclination data, 2s-compliment (X and Y channels)
    bit[11:0] = temperature data, unsigned              (temperature channel)
    '''

    def __init__(self) -> None:
        #self.errorFlag : np.typing.NDArray[np.int16] = np.array([], dtype=np.int16)
        #self.oldFlag : np.typing.NDArray[np.int16] = np.array([], dtype=np.int16)
        pass

    def convert(self, v:np.typing.NDArray[np.int16]) -> np.typing.NDArray[np.float64]:
        u = v.astype('>u2') # unsigned 16 bit integer
        b = np.bitwise_and(u, 1<<15) // (1<<15) # b==1: new data b==0: old data
        #self.oldFlag = np.where(b==0)[0]
        b = np.bitwise_and(u, 1<<14) // (1<<14) # b==1: error b==0: no error
        #self.errorFlag = np.where(b==1)[0]
        b = np.bitwise_and(u, (1<<13) + (1<<12))
        v_raw: np.typing.NDArray[np.int16] | np.typing.NDArray[np.uint16]
        if np.all(b == 0):
            #dtype = '>u2' # big endian unsigned int 2 byte (16 bit)
            v_raw = np.bitwise_and(u, (1<<13)-1)
            v_raw -= np.bitwise_and(u, 1<<13) # two's compliment.
            v_raw = v_raw.astype(np.uint16)
        else:
            #dtype = '>i2' # big endian signed int 2 byte (16 bit)
            v_raw = np.bitwise_and(u, (1<<13)-1)
            v_raw -= np.bitwise_and(u, 1<<13) # two's compliment.
            v_raw = v_raw.astype(np.int16)
        return v_raw.astype(np.float64)


class Deconvolve(object):
    r''' Description
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
  clarity. The fall-rate is about $\SI{0.17}{\m\per\s}$.'''

    def __init__(self,
                 X_dX:np.typing.NDArray[np.float64],
                 X:np.typing.NDArray[np.float64],
                 fs: float,
                 diff_gain: float):
        self.fs = fs
        self.diff_gain = diff_gain
        X = self.interpolate(X_dX, X)
        self.X_hires = self.bw_filter(X_dX, X)
        self.X = X


    def bw_filter(self,
                  X_dX:np.typing.NDArray[np.float64],
                  X:np.typing.NDArray[np.float64]
                  ) -> np.typing.NDArray[np.float64]:
        fc = 1/(2*np.pi * self.diff_gain)
        b, a = ss.butter(1, fc/(self.fs/2))
        if not X is None:
            p = np.polyfit(X, X_dX, 1)
            if p[0] < -0.5:
                X_dX = - X_dX
            z = ss.lfiltic(b, a, X[:1], X_dX[:1])
        else:
            tm = np.arange(0, 2*self.diff_gain, 1/self.fs)
            p = np.polyfit(tm, X_dX[:tm.shape[0]], 1)
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
                           

        
    def interpolate(self,
                    X_dX:np.typing.NDArray[np.float64],
                    X:np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        # only interpolate if X and X_dX are not of equal length
        if X is None or (X.shape == X_dX.shape):
            return X # nothing to do
        f_slow = self.fs * X.shape[0] / X_dX.shape[0]
        t_slow = np.arange(X.shape[0]) / f_slow
        t_fast = np.arange(X_dX.shape[0]) / self.fs
        ifun = si_PchipInterpolator(t_slow, X, extrapolate=True)
        return ifun(t_fast)





    
# ConverterDict=defaultdict(lambda : PassThrough,
#                           piezo=Piezo,
#                           gnd=Gnd,
#                           therm=Therm,
#                           shear=Shear,
#                           poly=Poly,
#                           voltage=Voltage,
#                           inclxy=InclXY,
#                           inclt=InclT,
#                           aem1g_a=Aem1g_a,
#                           none=PassThrough)
    
def get_converter(channel_config: common.ChannelConfigABC) -> type[Converter]:
    '''Converter factory

    Parameters
    ----------
    channel_config : ChannelConfig
        channel configuration dataclass

    Returns
    -------
    Converter object, relevant for the requested channel type.

    Note : this code uses the match/case structure and requires therefore python 3.10+
    '''
    converter : type[Converter]
    try:
        sensor_type = channel_config.type
    except KeyError:
        sensor_type = "none"
    match sensor_type:
        case "piezo":
            converter = Piezo
        case "gnd":
            converter = Gnd
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
        case "none":
            converter = PassThrough
        case _:
            converter = PassThrough

    return converter
