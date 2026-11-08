from pydantic import Field

from turban.instruments.generic.channels import (
    DefaultFloat,
    DefaultStr,
    DefaultInt,
    ChannelConfigBaseModel,
)


class ChannelConfig(ChannelConfigBaseModel):
    pass


class ChannelConfigPiezo(ChannelConfigBaseModel):
    a0: float = Field(default=DefaultFloat)


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


class ChannelConfigThermistorPreEmphasis(ChannelConfigBaseModel):
    diff_gain: float = Field(default=DefaultFloat)


class ChannelConfigShear(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: int = Field(default=DefaultInt)
    adc_zero: float = Field(default=DefaultFloat)
    sig_zero: float = Field(default=DefaultFloat)
    diff_gain: float = Field(default=DefaultFloat)
    sens: float = Field(default=DefaultFloat)
    sn: str = Field(default=DefaultStr)
    cal_date: str = Field(default=DefaultStr)


class ChannelConfigPressure(ChannelConfigBaseModel):
    coef0: float = Field(default=DefaultFloat)
    coef1: float = Field(default=DefaultFloat)
    coef2: float = Field(default=DefaultFloat)
    coef3: float = Field(default=DefaultFloat)
    cal_date: str = Field(default=DefaultStr)


class ChannelConfigPressurePreEmphasis(ChannelConfigBaseModel):
    diff_gain: float = Field(default=DefaultFloat)


class ChannelConfigPressureVoltage(ChannelConfigBaseModel):
    coef0: float = Field(default=DefaultFloat)
    coef1: float = Field(default=DefaultFloat)
    coef2: float = Field(default=DefaultFloat)


class ChannelConfigGnd(ChannelConfigBaseModel):
    coef0: float = Field(default=DefaultFloat)


class ChannelConfigVoltage(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: int = Field(default=DefaultInt)
    adc_zero: float = Field(default=DefaultFloat)
    g: float = Field(default=DefaultFloat)


class ChannelConfigInclinometer(ChannelConfigBaseModel):
    coef0: float = Field(default=DefaultFloat)
    coef1: float = Field(default=DefaultFloat)


class ChannelConfigEMC_CUR(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: int = Field(default=DefaultInt)
    adc_zero: float = Field(default=DefaultFloat)
    g: float = Field(default=DefaultFloat)


class ChannelConfigU_EM(ChannelConfigBaseModel):
    adc_fs: float = Field(default=DefaultFloat)
    adc_bits: int = Field(default=DefaultInt)
    adc_zero: float = Field(default=DefaultFloat)
    a: float = Field(default=DefaultFloat)
    b: float = Field(default=DefaultFloat)
    bias: float = Field(default=DefaultFloat)
    sn: str = Field(default=DefaultStr)
    cal_date: str = Field(default=DefaultStr)


# Maps MicroRider channel names (as used in the setup file) to their config class.
CHANNEL_CONFIGS: dict[str, type[ChannelConfigBaseModel]] = {
    "Gnd_2": ChannelConfig,
    "ch255": ChannelConfig,
    "Ax": ChannelConfigPiezo,
    "Ay": ChannelConfigPiezo,
    "T1": ChannelConfigThermistor,
    "T2": ChannelConfigThermistor,
    "T1_dT1": ChannelConfigThermistorPreEmphasis,
    "T2_dT2": ChannelConfigThermistorPreEmphasis,
    "sh1": ChannelConfigShear,
    "sh2": ChannelConfigShear,
    "P": ChannelConfigPressure,
    "P_dP": ChannelConfigPressurePreEmphasis,
    "PV": ChannelConfigPressureVoltage,
    "Gnd": ChannelConfigGnd,
    "V_Bat": ChannelConfigVoltage,
    "Incl_Y": ChannelConfigInclinometer,
    "Incl_X": ChannelConfigInclinometer,
    "Incl_T": ChannelConfigInclinometer,
    "EMC_Cur": ChannelConfigEMC_CUR,
    "U_EM": ChannelConfigU_EM,
}


def channel_config_factory(name: str) -> ChannelConfigBaseModel:
    """Return a fresh config object of the right class for channel `name`."""
    try:
        cls = CHANNEL_CONFIGS[name]
    except KeyError:
        raise ValueError(f"{name} is not a known MicroRider channel.") from None
    return cls(name=name)
