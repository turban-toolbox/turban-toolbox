from abc import abstractmethod, ABC
from turban.instruments.config import InstrumentConfig
from turban.process.shear.api import ShearLevel1


class Instrument(ABC):
    def __init__(self, cfg: InstrumentConfig):
        self.cfg = cfg


class Dropsonde(Instrument):
    @abstractmethod
    def to_shear_level1(self) -> "ShearLevel1":
        """
        Convert raw data to shear level 1
        """
        raise NotImplementedError("to_shear_level1 must be implemented in subclasses")
