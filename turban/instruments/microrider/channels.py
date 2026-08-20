from pydantic import Field

from turban.instruments.generic.channels import (
    DefaultFloat,
    DefaultStr,
    DefaultInt,
    ChannelConfigBaseModel,
    register_channel_config,
    channel_config_factory,
)
from turban.instruments.generic.api import InstrumentEnum    

# from typing import Any, Self, TypeVar
# from collections.abc import Callable
# from pydantic import BaseModel, Field, PrivateAttr


def microrider_channel_config_factory(name: str) -> ChannelConfigBaseModel:
    return channel_config_factory(name, InstrumentEnum.MicroRider)



@register_channel_config(InstrumentEnum.MicroRider, ["Gnd_2", "ch255"])
class ChannelConfig(ChannelConfigBaseModel):
    pass


@register_channel_config(InstrumentEnum.MicroRider, ["Ax", "Ay"])
class ChannelConfigPiezo(ChannelConfigBaseModel):
    a0: float = Field(default=DefaultFloat)


@register_channel_config(InstrumentEnum.MicroRider, ["T1", "T2"])
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


@register_channel_config(InstrumentEnum.MicroRider, ["T1_dT1", "T2_dT2"])
class ChannelConfigThermistorPreEmphasis(ChannelConfigBaseModel):
    diff_gain: float = Field(default=DefaultFloat)


@register_channel_config(InstrumentEnum.MicroRider, ["sh1", "sh2"])
class ChannelConfigShear(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: int = Field(default=DefaultInt)
    adc_zero: float = Field(default=DefaultFloat)
    sig_zero: float = Field(default=DefaultFloat)
    diff_gain: float = Field(default=DefaultFloat)
    sens: float = Field(default=DefaultFloat)
    sn: str = Field(default=DefaultStr)
    cal_date: str = Field(default=DefaultStr)


@register_channel_config(InstrumentEnum.MicroRider, ["P"])
class ChannelConfigPressure(ChannelConfigBaseModel):
    coef0: float = Field(default=DefaultFloat)
    coef1: float = Field(default=DefaultFloat)
    coef2: float = Field(default=DefaultFloat)
    coef3: float = Field(default=DefaultFloat)
    cal_date: str = Field(default=DefaultStr)


@register_channel_config(InstrumentEnum.MicroRider, ["P_dP"])
class ChannelConfigPressurePreEmphasis(ChannelConfigBaseModel):
    diff_gain: float = Field(default=DefaultFloat)


@register_channel_config(InstrumentEnum.MicroRider, ["PV"])
class ChannelConfigPressureVoltage(ChannelConfigBaseModel):
    coef0: float = Field(default=DefaultFloat)
    coef1: float = Field(default=DefaultFloat)
    coef2: float = Field(default=DefaultFloat)


@register_channel_config(InstrumentEnum.MicroRider, ["Gnd"])
class ChannelConfigGnd(ChannelConfigBaseModel):
    coef0: float = Field(default=DefaultFloat)


@register_channel_config(InstrumentEnum.MicroRider, ["V_Bat"])
class ChannelConfigVoltage(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: float = Field(default=DefaultFloat)
    adc_zero: float = Field(default=DefaultFloat)
    g: float = Field(default=DefaultFloat)


@register_channel_config(InstrumentEnum.MicroRider, ["Incl_Y", "Incl_X", "Incl_T"])
class ChannelConfigInclinometer(ChannelConfigBaseModel):
    coef0: float = Field(default=DefaultFloat)
    coef1: float = Field(default=DefaultFloat)


@register_channel_config(InstrumentEnum.MicroRider, ["EMC_Cur"])
class ChannelConfigEMC_CUR(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: float = Field(default=DefaultFloat)
    adc_zero: float = Field(default=DefaultFloat)
    g: float = Field(default=DefaultFloat)


@register_channel_config(InstrumentEnum.MicroRider, ["U_EM"])
class ChannelConfigU_EM(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: float = Field(default=DefaultFloat)
    adc_zero: float = Field(default=DefaultFloat)
    a: float = Field(default=DefaultFloat)
    b: float = Field(default=DefaultFloat)
    bias: float = Field(default=DefaultFloat)
    sn: str = Field(default=DefaultStr)
    cal_date: str = Field(default=DefaultStr)


def channel_config_factory(name: str, prefix: str) -> ChannelConfigBaseModel:
    key = f"{prefix}{name}"
    if key not in _CHANNEL_CONFIG_REGISTRY:
        raise ValueError(f"{key} is not a valid channel name.")
    return _CHANNEL_CONFIG_REGISTRY[key](name=name)

def microrider_channel_config_factory(name: str) -> ChannelConfigBaseModel:
    return channel_config_factory(name, InstrumentEnum.MicroRider)
