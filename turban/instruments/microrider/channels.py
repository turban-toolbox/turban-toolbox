from turban.instruments.generic.channels import ChannelConfigBaseModel


class ChannelConfig(ChannelConfigBaseModel):
    pass


class ChannelConfigPiezo(ChannelConfigBaseModel):
    a0: float = 0.0


class ChannelConfigThermistor(ChannelConfigBaseModel):
    adc_fs: float
    adc_bits: int
    a: float
    b: float
    g: float
    e_b: float
    sn: str = ""
    beta: float | None = None
    beta_1: float | None = None
    beta_2: float | None = None
    beta_3: float | None = None
    t_0: float
    cal_date: str = ""


class ChannelConfigThermistorPreEmphasis(ChannelConfigBaseModel):
    diff_gain: float


class ChannelConfigShear(ChannelConfigBaseModel):
    adc_fs: float
    adc_bits: int
    adc_zero: float = 0.0
    sig_zero: float = 0.0
    diff_gain: float
    sens: float
    sn: str = ""
    cal_date: str = ""


class ChannelConfigPressure(ChannelConfigBaseModel):
    coef0: float
    coef1: float | None = None
    coef2: float | None = None
    coef3: float | None = None
    cal_date: str = ""


class ChannelConfigPressurePreEmphasis(ChannelConfigBaseModel):
    diff_gain: float


class ChannelConfigPressureVoltage(ChannelConfigBaseModel):
    coef0: float
    coef1: float | None = None
    coef2: float | None = None


class ChannelConfigGnd(ChannelConfigBaseModel):
    coef0: float | None = None  # not used


class ChannelConfigVoltage(ChannelConfigBaseModel):
    adc_fs: float
    adc_bits: int
    adc_zero: float = 0.0
    g: float = 1.0


class ChannelConfigInclinometer(ChannelConfigBaseModel):
    coef0: float
    coef1: float


class ChannelConfigInclinometerT(ChannelConfigInclinometer):
    coef0: float = 624.0
    coef1: float = -0.47


class ChannelConfigEMC_CUR(ChannelConfigBaseModel):
    adc_fs: float
    adc_bits: int
    adc_zero: float = 0.0
    g: float = 1.0


class ChannelConfigU_EM(ChannelConfigBaseModel):
    adc_fs: float | None = None  # analog (aem1g_a) only
    adc_bits: int | None = None  # analog (aem1g_a) only
    adc_zero: float | None = None  # analog: defaults to adc_fs / 2
    a: float
    b: float
    bias: float = 0.0
    sn: str = ""
    cal_date: str = ""


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
    "Incl_T": ChannelConfigInclinometerT,
    "EMC_Cur": ChannelConfigEMC_CUR,
    "U_EM": ChannelConfigU_EM,
}
