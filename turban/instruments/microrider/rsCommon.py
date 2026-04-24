from dataclasses import dataclass
from numpy.typing import NDArray
from numpy import float64
from typing import Any, Self, TypeVar
from collections.abc import Callable
from pydantic import BaseModel, Field, PrivateAttr


@dataclass(kw_only=True)
class ByteHeader:
    file_number: int
    record_number: int
    record_number_serial_port: int
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    millisecond: int
    header_version: float
    setupfile_size: int
    product_ID: int
    build_number: int
    timezone_in_minutes: int
    buffer_status: int
    restarted: int
    record_header_size: int
    data_record_size: int
    number_of_records_written: int
    frequency_clock: float
    fast_cols: int
    slow_cols: int
    n_rows: int
    data_size: int


@dataclass(kw_only=True)
class Header:
    full_path: str
    n_cols: int
    n_records: int
    fs_fast: float
    fs_slow: float
    header_version: float
    matrix_count: int
    t_slow: NDArray[float64]
    t_fast: NDArray[float64]
    timestamp: float
    date: str
    time: str


### Data classes for all channels

DefaultFloat = -9999999999999.0
DefaultStr = "undefined"
DefaultInt = -999999999999


class ChannelConfigBaseModel(BaseModel):
    name: str = ""
    id: int = 0
    type: str = "unknown"
    sign: str = "signed"
    sample_rate: float = 0.0
    units: str = ""

    _set_properties: list[str] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if (
                (type(value) == float and value != DefaultFloat)
                or (type(value) == int and value != DefaultInt)
                or (type(value) == str and value != DefaultStr)
            ):
                self._set_properties.append(field_name)

    def __repr__(self) -> str:
        s: str = f"{self.__class__.__name__}:\n"
        for k in self._set_properties:
            s += f"    {k}: {getattr(self, k)}\n"
        return s

    def update(self, k: str, v: Any) -> None:
        if not hasattr(self, k):
            raise AttributeError(f"{self.__class__.__name__} has no attribute {k}.")
        setattr(self, k, v)
        if k not in self._set_properties:
            self._set_properties.append(k)

    def is_set(self, k: str) -> bool:
        if not hasattr(self, k):
            raise AttributeError(f"{self.__class__.__name__} has no attribute {k}.")
        return k in self._set_properties

    def clone(self) -> Self:
        new = self.model_copy()
        object.__setattr__(new, "_set_properties", list(self._set_properties))
        return new


# Create a registry of ChannelConfig data classes:

T = TypeVar(
    "T", bound=ChannelConfigBaseModel
)  # A type machting also all its derivates.
_CHANNEL_CONFIG_REGISTRY: dict[str, type[ChannelConfigBaseModel]] = {}


def register_channel_config(names: list[str]) -> Callable[[type[T]], type[T]]:
    def wrapper(cls: type[T]) -> type[T]:
        for name in names:
            _CHANNEL_CONFIG_REGISTRY[name] = cls
        return cls

    return wrapper


@register_channel_config(["Gnd_2"])
class ChannelConfig(ChannelConfigBaseModel):
    pass


@register_channel_config(["Ax", "Ay"])
class ChannelConfigPiezo(ChannelConfigBaseModel):
    a0: float = Field(default=DefaultFloat)


@register_channel_config(["T1", "T2"])
class ChannelConfigThermistor(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: int = Field(default=DefaultInt)
    a: float = Field(default=DefaultFloat)
    b: float = Field(default=DefaultFloat)
    g: float = Field(default=DefaultFloat)
    e_b: float = Field(default=DefaultFloat)
    sn: str = Field(default=DefaultStr)
    beta: float = Field(default=DefaultFloat)
    beta_1: float = Field(default=DefaultFloat)
    beta_2: float = Field(default=DefaultFloat)
    beta_3: float = Field(default=DefaultFloat)
    t_0: float = Field(default=DefaultFloat)
    cal_date: str = Field(default=DefaultStr)


@register_channel_config(["T1_dT1", "T2_dT2"])
class ChannelConfigThermistorPreEmphasis(ChannelConfigBaseModel):
    diff_gain: float = Field(default=DefaultFloat)


@register_channel_config(["sh1", "sh2"])
class ChannelConfigShear(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: int = Field(default=DefaultInt)
    adc_zero: float = Field(default=DefaultFloat)
    sig_zero: float = Field(default=DefaultFloat)
    diff_gain: float = Field(default=DefaultFloat)
    sens: float = Field(default=DefaultFloat)
    sn: str = Field(default=DefaultStr)
    cal_date: str = Field(default=DefaultStr)


@register_channel_config(["P"])
class ChannelConfigPressure(ChannelConfigBaseModel):
    coef0: float = Field(default=DefaultFloat)
    coef1: float = Field(default=DefaultFloat)
    coef2: float = Field(default=DefaultFloat)
    coef3: float = Field(default=DefaultFloat)
    cal_date: str = Field(default=DefaultStr)


@register_channel_config(["P_dP"])
class ChannelConfigPressurePreEmphasis(ChannelConfigBaseModel):
    diff_gain: float = Field(default=DefaultFloat)


@register_channel_config(["PV"])
class ChannelConfigPressureVoltage(ChannelConfigBaseModel):
    coef0: float = Field(default=DefaultFloat)
    coef1: float = Field(default=DefaultFloat)
    coef2: float = Field(default=DefaultFloat)


@register_channel_config(["Gnd"])
class ChannelConfigGnd(ChannelConfigBaseModel):
    coef0: float = Field(default=DefaultFloat)


@register_channel_config(["V_Bat"])
class ChannelConfigVoltage(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: float = Field(default=DefaultFloat)
    adc_zero: float = Field(default=DefaultFloat)
    g: float = Field(default=DefaultFloat)


@register_channel_config(["Incl_Y", "Incl_X", "Incl_T"])
class ChannelConfigInclinometer(ChannelConfigBaseModel):
    coef0: float = Field(default=DefaultFloat)
    coef1: float = Field(default=DefaultFloat)


@register_channel_config(["EMC_Cur"])
class ChannelConfigEMC_CUR(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: float = Field(default=DefaultFloat)
    adc_zero: float = Field(default=DefaultFloat)
    g: float = Field(default=DefaultFloat)


@register_channel_config(["U_EM"])
class ChannelConfigU_EM(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: float = Field(default=DefaultFloat)
    adc_zero: float = Field(default=DefaultFloat)
    a: float = Field(default=DefaultFloat)
    b: float = Field(default=DefaultFloat)
    bias: float = Field(default=DefaultFloat)
    sn: str = Field(default=DefaultStr)
    cal_date: str = Field(default=DefaultStr)


def channel_config_factory(name: str) -> type[ChannelConfigBaseModel]:
    if name not in _CHANNEL_CONFIG_REGISTRY:
        raise ValueError(f"{name} is not a valid channel name.")
    return _CHANNEL_CONFIG_REGISTRY[name]
