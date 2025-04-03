from abc import abstractmethod
from turban.instruments.config import InstrumentConfig
from turban.shear import ShearLevel1


class Instrument:
    def __init__(self, cfg: InstrumentConfig):
        self.cfg = cfg


class Dropsonde(Instrument):
    @abstractmethod
    def to_shear_level1(self) -> "ShearLevel1":
        """
        Convert raw data to shear level 1
        """
        raise NotImplementedError("to_shear_level1 must be implemented in subclasses")
