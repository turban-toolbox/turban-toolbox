from typing import Literal, Union, Optional, Annotated
import logging
import sys
from typing import cast
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field

from . import mss_mrd

from turban.process.shear.config import ShearConfig
from turban.process.shear.api import ShearLevel1

# Setup logging module
# TODO should handle this more gracefully, having debug level logging everywhere is annoying
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger("turban.instruments.mss")
# logger.setLevel(logging.DEBUG)


# Define standard names for sensors
mss_standard_ctd_sensornames = {}
mss_standard_ctd_sensornames["press"] = ["PRESS", "P250", "P1000"]
mss_standard_ctd_sensornames["temp"] = ["TEMP", "NTC"]
mss_standard_ctd_sensornames["cond"] = ["COND"]


class MssSensor(BaseModel):
    name: str
    coefficients: list[float]
    channel: int
    unit: str = Field(default="")
    calibration_type: Literal[None]  # ["N", "SHE", "P", "SHH", "NFC", "V04", "N24"]


class MssSensorPoly(MssSensor):
    calibration_type: Literal["N"] = Field(default="N")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._p = np.polynomial.Polynomial(self.coefficients)

    def raw_to_units(self, rawdata, offset=0):
        data = self._p(rawdata + offset)
        return data


class MssShearSensor(MssSensor):
    sensitivity: float
    serial_number: str = Field(default="")
    reference_temperature: float = Field(default=-9999)
    calibration_date: str = Field(default="")
    calibration_type: Literal["SHE"] = Field(default="SHE")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coefficients = [None, None]
        self.coefficients[0] = 1.47133e-6 / self.sensitivity
        self.coefficients[1] = 2.94266e-6 / self.sensitivity
        self._p = np.polynomial.Polynomial(self.coefficients)

    def raw_to_units(self, rawdata, offset=0):
        data = self._p(rawdata - offset)  # The shear sensors have the negative offset
        return data


class MssSensorPressure(MssSensor):
    calibration_type: Literal["P"] = Field(default="P")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._p = np.polynomial.Polynomial(self.coefficients[:-1])

    def raw_to_units(self, rawdata, offset=0):
        data = self._p(rawdata + offset) - self.coefficients[-1]
        return data


class MssSensorNTC(MssSensor):
    """
    Steinhart/Hart NTC Polynomial
    """

    calibration_type: Literal["SHH"] = Field(default="SHH")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._p = np.polynomial.Polynomial(self.coefficients[:-1])

    def raw_to_units(self, rawdata, offset):
        data = self._p(np.log(rawdata + offset))
        data = 1 / data - 273.15  # Kelvin to degC
        return data


class MssSensorTurb(MssSensor):
    """
    Convert rawdata turbidity to NFC
    """

    calibration_type: Literal["NFC"] = Field(default="NFC")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._p = np.polynomial.Polynomial(self.coefficients[:-1])

    def raw_to_units(self, rawdata, offset):
        p = np.polynomial.Polynomial(self.coefficients[:-2])
        data = p(rawdata + offset) * self.coefficients[-1] + self.coefficients[-2]
        return data


class MssSensorOptode(MssSensor):
    """
    Convert oxygen optode rawdata to physical units
    """

    calibration_type: Literal["V04"] = Field(default="V04")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._p1 = np.polynomial.Polynomial(
            self.coefficients[0:2]
        )  # Convert data to mV
        self._p2 = np.polynomial.Polynomial(
            self.coefficients[-2:]
        )  # 0 Point correction with B0 and B1

    def raw_to_units(self, rawdata, offset):
        data_mV = self._p1(rawdata + offset)
        data = self._p2(data_mV)
        return data


class MssSensorOptodeInternalTemp(MssSensor):
    calibration_type: Literal["N24"] = Field(default="N24")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._p = np.polynomial.Polynomial(self.coefficients)

    def raw_to_units(self, rawdata, offset=0):
        data = self._p(rawdata + offset)
        return data


class MssDeviceConfig(BaseModel):
    offset: int = Field(
        default=0, description="16bit offset, typically 0, older devices have -32768"
    )
    sampling_freq: float = Field(
        default=1024,
        description="The sampling frequency [Hz] of the microstructure probe",
    )
    sensors: dict[
        str,
        Annotated[
            Union[
                MssSensor,
                MssSensorPoly,
                MssShearSensor,
                MssSensorPressure,
                MssSensorNTC,
                MssSensorTurb,
                MssSensorOptode,
                MssSensorOptodeInternalTemp,
            ],
            Field(discriminator="calibration_type"),
        ],
    ] = Field(
        default={}, description="A dictionary of the sensors mounted to the probe"
    )
    sensornames_ctd: dict[
        Union[Literal["cond"], Literal["temp"], Literal["press"]],
        str,
    ] = Field(
        default={"cond": "", "temp": "", "press": ""},
        description="A dictionary to link standard ctd names to the names of the MSS config",
    )
    pressure_sensorname: Optional[str] = Field(
        default=None,
        description="The sensorname of the pressure sensor, if None a best guess will be made",
    )
    pspd_rel_method: Literal["pressure", "constant", "external"] = Field(
        default="pressure",
        description="Method for the platform speed relative to the seawater, this is needed to calculate wavenumbers from the sampled data",
    )
    pspd_rel_constant_vel: Optional[float] = Field(
        default=None,
        description='Constant velocity [m/s] used as pspd_rel, if defined by "pspd_rel_method"',
    )

    @classmethod
    def from_mrd(
        cls,
        filename: str,
        shear_sensitivities: dict[str, float],
        offset: int = 0,
    ):
        """
        Creating a MssDeviceConfig from a mrd file
        """
        self = cls()
        logger.debug("Opening file:{}".format(filename))
        mrd_file = open(filename, "rb")
        data = mss_mrd.read_mrd(filestream=mrd_file, header_only=True)
        logger.debug("Closing file:{}".format(filename))
        mrd_file.close()
        header_raw = data["header"]
        header = mss_mrd.parse_header(header_raw)
        # Check for CTD sensors and link names
        for ctd_sensor in self.sensornames_ctd.keys():
            sensorname_mss = self.sensornames_ctd[ctd_sensor]
            if len(sensorname_mss) == 0:
                logger.debug("Searching for sensor of {}".format(ctd_sensor))
                sensornames = [
                    header["mss"]["channels"][ch]["name"]
                    for ch in header["mss"]["channels"]
                ]
                sensornames_casefold = [
                    header["mss"]["channels"][ch]["name"].casefold()
                    for ch in header["mss"]["channels"]
                ]
                for k in mss_standard_ctd_sensornames[
                    ctd_sensor
                ]:  # Loop over the stanndard names
                    if k.casefold() in sensornames_casefold:
                        index_sensor = sensornames_casefold.index(k.casefold())
                        sensorname_mss = sensornames[index_sensor]
                        logger.debug(
                            "\tFound MSS sensor {} for {}".format(
                                sensorname_mss, ctd_sensor
                            )
                        )
                        self.sensornames_ctd[ctd_sensor] = sensorname_mss
                        break

        # Fill in sensors from header
        for ch in header["mss"]["channels"]:
            sensor_dict = header["mss"]["channels"][ch]
            sensorname = sensor_dict["name"]
            unit = sensor_dict["unit"]
            caltype = sensor_dict["caltype"].upper()
            logger.debug(
                "Checking Channel:{}, sensorname:{}, caltype:{}".format(
                    ch, sensorname, caltype
                )
            )
            if caltype == "N":  # Polynom
                if sensorname.upper().startswith("SHE"):
                    logger.debug("\tAdding shear sensor {}".format(sensorname))
                    sensitivity = shear_sensitivities[sensorname]
                    self.sensors[sensorname] = MssShearSensor(
                        channel=ch,
                        name=sensorname,
                        coefficients=sensor_dict["coeff"],
                        unit=unit,
                        sensitivity=sensitivity,
                    )
                else:
                    logger.debug(
                        "\tAdding standard polynomial sensor {}".format(sensorname)
                    )
                    self.sensors[sensorname] = MssSensorPoly(
                        channel=ch,
                        name=sensorname,
                        coefficients=sensor_dict["coeff"],
                        unit=unit,
                    )
            elif caltype == "SHH":
                logger.debug("\tAdding NTC sensor {}".format(sensorname))
                self.sensors[sensorname] = MssSensorNTC(
                    channel=ch,
                    name=sensorname,
                    coefficients=sensor_dict["coeff"],
                    unit=unit,
                )
            elif caltype == "P":  # Pressure
                logger.debug("\tAdding pressure sensor {}".format(sensorname))
                self.sensors[sensorname] = MssSensorPressure(
                    channel=ch,
                    name=sensorname,
                    coefficients=sensor_dict["coeff"],
                    unit=unit,
                )
            elif caltype == "V04":  # Oxygen
                logger.debug("\tAdding oxygen optode sensor {}".format(sensorname))
                self.sensors[sensorname] = MssSensorOptode(
                    channel=ch,
                    name=sensorname,
                    coefficients=sensor_dict["coeff"],
                    unit=unit,
                )
            elif (
                caltype == "N24"
            ):  # Internal temperature of oxygensensor
                logger.debug(
                    "\tAdding oxygen optode temperature sensor {}".format(sensorname)
                )
                self.sensors[sensorname] = MssSensorOptodeInternalTemp(
                    channel=ch,
                    name=sensorname,
                    coefficients=sensor_dict["coeff"],
                    unit=unit,
                )
            elif caltype == "NFC":  # Turbidity etc.
                logger.debug("\tAdding NFC sensor {}".format(sensorname))
                self.sensors[sensorname] = MssSensorTurb(
                    channel=ch,
                    name=sensorname,
                    coefficients=sensor_dict["coeff"],
                    unit=unit,
                )
        # print('Header', header['channels'])

        return self

    @classmethod
    def from_prb(cls, filename, shear_sensitivities=[None, None], offset=0):
        """
        Creating a MssDeviceConfig from a prb file
        """
        raise NotImplementedError


def mrd_to_shear_level1(
    fpath: Path,
    shear_config: ShearConfig,
    mss_config: MssDeviceConfig | None = None,
    shear_sensitivities: dict[str, float] | None = None,
):
    if mss_config is None:
        # in this case, shear_sensitivities must not be None
        shear_sensitivities = cast(dict[str, float], shear_sensitivities)
        mss_config = MssDeviceConfig.from_mrd(
            filename=str(fpath),
            shear_sensitivities=shear_sensitivities,
            offset=0,
        )

    with open(fpath, "rb") as f:
        data_raw = mss_mrd.read_mrd(f)

    data_level0 = mss_mrd.raw_to_level0(mss_config, data_raw)
    data_level1 = mss_mrd.level0_to_level1(mss_config, data_level0)

    return ShearLevel1(
        time=np.asarray(data_level1["time_count"]),
        senspeed=np.asarray(data_level1["PSPD_REL"]),
        shear=np.asarray(data_level1["SHEAR"]),
        section_number=np.zeros_like(data_level1["time_count"], dtype=int),
        cfg=shear_config,
    )

