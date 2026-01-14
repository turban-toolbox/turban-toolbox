from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field, fields
from numpy.typing import NDArray
from numpy import float64
from typing import Any, Self, Type, TypeVar, Callable


@dataclass(kw_only=True)
class ByteHeader:
    file_number: int
    record_number: int
    record_number_serial_port: int
    year : int
    month : int
    day : int
    hour : int
    minute: int
    second : int
    millisecond : int
    header_version : float
    setupfile_size : int
    product_ID : int 
    build_number : int 
    timezone_in_minutes : int
    buffer_status : int
    restarted : int
    record_header_size : int
    data_record_size : int
    number_of_records_written : int 
    frequency_clock : float
    fast_cols : int
    slow_cols : int
    n_rows : int
    data_size : int

@dataclass(kw_only=True)
class Header:
    full_path : str
    n_cols : int
    n_records : int
    fs_fast : float
    fs_slow : float
    header_version : float
    matrix_count : int
    t_slow : NDArray[float64]
    t_fast : NDArray[float64]
    timestamp : float
    date : str
    time : str
    


### Data classes for all channels
    
DefaultFloat = -9999999999999.
DefaultStr = "undefined"
DefaultInt = -999999999999

@dataclass
class ChannelConfigABC:
    name: str = ""
    id : int = 0
    type :str = "unknown"
    sign : str = "signed"
    sample_rate : float = 0.
    units : str = ""

    # marks all fields given to the constructor as "set"
    def __post_init__(self) -> None:
        cls_fields = fields(self.__class__)
        self._set_properties : list[str] = []
        for field in cls_fields:
            value = getattr(self, field.name)
            if (type(value) == float and value!=DefaultFloat) or \
               (type(value) == int and value!=DefaultInt) or \
               (type(value) == str and value!=DefaultStr) :
                self._set_properties.append(field.name)

    def __repr__(self) -> str:
        s : str = f"{self.__class__.__name__}:\n"
        for k in self._set_properties:
            s += f"    {k}: {getattr(self, k)}\n"
        return s
        
    def update(self, k : str, v : Any) -> None:
        if not hasattr(self, k):
            raise AttributeError(f"{self.__class__.__name__} has no attribute {k}.")
        setattr(self, k, v)
        if k not in self._set_properties:
            self._set_properties.append(k)

    def is_set(self, k: str) -> bool:
        if not hasattr(self, k):
            raise AttributeError(f"{self.__class__.__name__} has no attribute {k}.")
        return k in self._set_properties

    def copy(self) -> Self:
        return copy.copy(self)



# Create a registry of ChannelConfig data classes:

T = TypeVar('T', bound=ChannelConfigABC) # A type machting also all its derivates.
_CHANNEL_CONFIG_REGISTRY : dict[str, Type[ChannelConfigABC]] = {}

def register_channel_config(names: list[str]) -> Callable[[Type[T]], Type[T]]:
    def wrapper(cls: Type[T]) -> Type[T]:
        for name in names:
            _CHANNEL_CONFIG_REGISTRY[name] = cls
        return cls
    return wrapper


@register_channel_config(["Gnd_2"])
@dataclass
class ChannelConfig(ChannelConfigABC):
    pass

@register_channel_config(["Ax", "Ay"])
@dataclass
class ChannelConfigPiezo(ChannelConfigABC):
    a0: float = field(default=DefaultFloat)

@register_channel_config(["T1", "T2"])
@dataclass
class ChannelConfigThermistor(ChannelConfigABC):
    adc_fs : float = field(default=DefaultFloat)
    adc_bits : int = field(default=DefaultInt)
    a : float = field(default=DefaultFloat)
    b : float = field(default=DefaultFloat)
    g : float = field(default=DefaultFloat)
    e_b : float = field(default=DefaultFloat)
    sn : str = field(default=DefaultStr)
    beta : float = field(default=DefaultFloat)
    beta_1 :float = field(default=DefaultFloat)
    beta_2 :float = field(default=DefaultFloat)
    beta_3 : float = field(default=DefaultFloat)
    t_0 : float = field(default=DefaultFloat)
    cal_date: str = field(default=DefaultStr)


@register_channel_config(["T1_dT1", "T2_dT2"])    
@dataclass
class ChannelConfigThermistorPreEmphasis(ChannelConfigABC):
    diff_gain : float = field(default=DefaultFloat)

@register_channel_config(["sh1", "sh2"])    
@dataclass    
class ChannelConfigShear(ChannelConfigABC):
    adc_fs : float = field(default=DefaultFloat)
    adc_bits : int = field(default=DefaultInt)
    adc_zero : float = field(default=DefaultFloat)
    sig_zero : float = field(default=DefaultFloat)
    diff_gain : float = field(default=DefaultFloat)
    sens : float = field(default=DefaultFloat)
    sn : str = field(default=DefaultStr)
    cal_date: str = field(default=DefaultStr)

@register_channel_config(["P"])    
@dataclass    
class ChannelConfigPressure(ChannelConfigABC):
    coef0 : float = field(default=DefaultFloat)
    coef1 : float = field(default=DefaultFloat)
    coef2 : float = field(default=DefaultFloat)
    coef3 : float = field(default=DefaultFloat)
    cal_date: str = field(default=DefaultStr)

@register_channel_config(["P_dP"])    
@dataclass    
class ChannelConfigPressurePreEmphasis(ChannelConfigABC):
    diff_gain : float = field(default=DefaultFloat)

@register_channel_config(["PV"])    
@dataclass    
class ChannelConfigPressureVoltage(ChannelConfigABC):
    coef0 : float = field(default=DefaultFloat)
    coef1 : float = field(default=DefaultFloat)
    coef2 : float = field(default=DefaultFloat)

@register_channel_config(["Gnd"])    
@dataclass
class ChannelConfigGnd(ChannelConfigABC):
    coef0 : float = field(default=DefaultFloat)

@register_channel_config(["V_Bat"])    
@dataclass
class ChannelConfigVoltage(ChannelConfigABC):
    adc_fs : float = field(default=DefaultFloat)
    adc_bits : float = field(default=DefaultFloat)
    adc_zero : float = field(default=DefaultFloat)
    g : float = field(default=DefaultFloat)

@register_channel_config(["Incl_Y", "Incl_X", "Incl_T"])    
@dataclass    
class ChannelConfigInclinometer(ChannelConfigABC):
    coef0 : float = field(default=DefaultFloat)
    coef1 : float = field(default=DefaultFloat)

@register_channel_config(["EMC_Cur"])    
@dataclass    
class ChannelConfigEMC_CUR(ChannelConfigABC):
    adc_fs : float = field(default=DefaultFloat)
    adc_bits : float = field(default=DefaultFloat)
    adc_zero : float = field(default=DefaultFloat)
    g : float = field(default=DefaultFloat)

@register_channel_config(["U_EM"])    
@dataclass    
class ChannelConfigU_EM(ChannelConfigABC):
    adc_fs : float = field(default=DefaultFloat)
    adc_bits : float = field(default=DefaultFloat)
    adc_zero : float = field(default=DefaultFloat)
    a : float = field(default=DefaultFloat)
    b : float = field(default=DefaultFloat)
    bias : float = field(default=DefaultFloat)
    sn : str = field(default=DefaultStr)
    cal_date: str = field(default=DefaultStr)


def channel_config_factory(name: str) -> Type[ChannelConfigABC]:
    if name not in _CHANNEL_CONFIG_REGISTRY:
        raise ValueError(f"{name} is not a valid channel name.")
    return _CHANNEL_CONFIG_REGISTRY[name]
