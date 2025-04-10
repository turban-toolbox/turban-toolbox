"""Defines high-level API to interact with TURBAN toolbox"""

from logging import warnings
from typing import get_type_hints
from abc import abstractmethod, ABC
from typing import Literal
from dataclasses import dataclass
from jaxtyping import Float, Int, AbstractArray, Num
from turban.config import SegmentConfig
from numpy import newaxis, nan, ndarray
import numpy as np
import xarray as xr

from turban.util import agg_fast_to_slow, fast_to_slow_reshape_index


@dataclass(kw_only=True)
class TimeseriesLevel:
    time: Float[ndarray, "time"]

    _coords = ["time"]

    def arrays_as_dict(self):
        return {
            name: (
                [dim.name for dim in t.dims],
                getattr(self, name),
            )
            for name, t in get_type_hints(self).items()
            if issubclass(t, AbstractArray)
        }

    def to_xarray(self):
        dct = self.arrays_as_dict()
        data_vars = {k: v for k, v in dct.items() if k not in self._coords}
        coords = {k: v for k, v in dct.items() if k in self._coords}
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def get_attr(self, name):
        attr = getattr(self, name)
        if isinstance(attr, SegmentConfig):
            return getattr()


@dataclass(kw_only=True)
class HasLevelBelow(TimeseriesLevel):
    level_below: TimeseriesLevel | None

    @classmethod
    @abstractmethod
    def from_level_below(cls, data: TimeseriesLevel | None):
        pass

    @property
    def cfg(self):
        if self.level_below is not None:
            return self.level_below.cfg
        elif hasattr(self, "_cfg"):
            return self._cfg
        else:
            raise ValueError

    @cfg.setter
    def cfg(self, value):
        if self.level_below is not None:
            self.level_below.cfg = value
        else:
            self._cfg = value


@dataclass(kw_only=True)
class Level1(TimeseriesLevel):
    pspd: Float[ndarray, "time"]
    cfg: SegmentConfig  # only define this here - other levels get it through HasLevelBelow


@dataclass(kw_only=True)
class Level2(HasLevelBelow, TimeseriesLevel):
    pspd: Float[ndarray, "time"]


@dataclass(kw_only=True)
class Level3(HasLevelBelow, TimeseriesLevel):
    waveno: Float[ndarray, "time waveno"]
    freq: Float[ndarray, "waveno"]
    platform_speed: Float[ndarray, "time"]

    _coords = ["time", "freq"]


@dataclass(kw_only=True)
class Level4(HasLevelBelow, TimeseriesLevel):
    pass


@dataclass(kw_only=True)
class AggAux:
    def __init__(
        self,
        slow: dict[str, Num[ndarray, "... time_slow"]],
        fast: dict[str, Num[ndarray, "... time_fast"]],
        diss_length: int, diss_overlap: int,
        section_marker: Int[ndarray, "... time_fast"] | None = None,
        
    ) -> None:
        
        self._data_slow = slow
        self._data_fast = fast
        if section_marker is None:
            self.section_marker
        self.section_marker = section_marker
        self._data_fast_len = section_marker.shape[-1]

    def agg(self,  agg_methods: dict[str, str]):
        if self._data_slow
        ii = fast_to_slow_reshape_index(data_len=)
        for agg_method, var_name in agg_methods.items():



class Processing(ABC):
    """Propagates a given start level up to level 4."""

    @property
    @abstractmethod
    def _level_mapping(self) -> dict:
        # Slightly clumsy way of requiring a class attribute called _level_mapping
        return {1: Level1, 2: Level2, 3: Level3, 4: Level4}

    @property
    @abstractmethod
    def _agg_mapping(self) -> dict:
        """Slightly clumsy way of requiring a class attribute called `_agg_mapping`.
        `_agg_mapping` determines which quantities are aggregated from fast to slow time.

        {agg_method: [name]}"""
        return {"mean": []}

    def __init__(self, data: TimeseriesLevel, level: Literal[1, 2, 3, 4]):
        for l in range(level + 1, 5):
            data = self._level_mapping[l].from_level_below(data)
        self.data = data

    @property
    def level1(self):
        if self.level2 is None:
            return None
        else:
            return self.level2.level_below

    @property
    def level2(self):
        if self.level3 is None:
            return None
        else:
            return self.level3.level_below

    @property
    def level3(self):
        if self.level4 is None:
            return None
        else:
            return self.level4.level_below

    @property
    def level4(self):
        return self.data

    @property
    def cfg(self):
        return self.level4.cfg
