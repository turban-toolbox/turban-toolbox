"""Defines high-level API to interact with TURBAN toolbox"""

from functools import wraps
from logging import warnings
from typing import get_type_hints
from abc import abstractmethod, ABC
from typing import Literal
from dataclasses import dataclass
from jaxtyping import Float, Int, AbstractArray, Num
from turban.process.generic.config import SegmentConfig
from numpy import newaxis, nan, ndarray
import numpy as np
import xarray as xr

from turban.utils.util import agg_fast_to_slow, get_chunking_index

_AuxDataTypehint = dict[
    str,
    tuple[
        list[str],
        Num[ndarray, "*any time_fast"],
        dict[str, str | None],
    ],
]


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


class AggAux:
    """Aggregates auxiliary variables from fast to slow timesteps"""

    def __init__(
        self,
        data_len: int,
        diss_length: int,
        diss_overlap: int,
        section_marker: Int[ndarray, "... data_len"] | None = None,
    ) -> None:
        if section_marker is None:
            self.section_marker = np.ones((data_len), dtype=int)
        else:
            self.section_marker = section_marker

        self._data_len = data_len
        self._diss_length = diss_length
        self._diss_overlap = diss_overlap
        self._agg_index = get_chunking_index(
            data_len,
            diss_length,
            0,
            diss_length,
            diss_overlap,
            section_marker,
        )

    def agg(
        self,
        data: _AuxDataTypehint,
        coords: list[str],
    ) -> None:
        """Aggregates data in the form
        {variable_name_fast: ([dims], ndarray, {agg_method: variable_new_slow}])}"""
        slow = {}
        for varname, (dims, arr, rename_dict) in data.items():
            for agg_method, varname_new in rename_dict.items():
                if varname_new is None:
                    varname_new = f"{varname}_{agg_method}"
                slow[varname_new] = (
                    dims,
                    agg_fast_to_slow(
                        arr, reshape_index=self._agg_index, agg_method=agg_method
                    ),
                )
        self._slow = slow
        self._fast = {varname: (dims, arr) for varname, (dims, arr, _) in data.items()}
        self._coords = coords

    def to_xarray(self):
        coords, data_vars = _split_dict_by(self._slow, self._coords)
        slow = xr.Dataset(data_vars=data_vars, coords=coords)
        coords, data_vars = _split_dict_by(self._fast, self._coords)
        fast = xr.Dataset(data_vars=data_vars, coords=coords)
        return slow, fast


class Processing(ABC):
    """Propagates a given start level up to level 4."""

    @property
    @abstractmethod
    def _level_mapping(self) -> dict:
        # Slightly clumsy way of requiring a class attribute called _level_mapping
        return {1: Level1, 2: Level2, 3: Level3, 4: Level4}

    def __init__(
        self,
        data: TimeseriesLevel,
        level: Literal[1, 2, 3, 4],
        data_aux: _AuxDataTypehint | None = None,
        coords_aux: list[str] | None = None,
        cls_aux: type = AggAux,
    ):
        for l in range(level + 1, 5):
            data = self._level_mapping[l].from_level_below(data)
        self.data = data
        agg = cls_aux(
            self.data_len_fast,
            self.cfg.diss_length,
            self.cfg.diss_overlap,
            self.level1.section_marker,
        )
        if data_aux is not None and coords_aux is not None:
            agg.agg(data_aux, coords_aux)
        self.aux = agg

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

    @property
    def data_len_fast(self):
        """Length of the fast time vector (levels 1 and 2)"""
        if self.level2 is None:
            return None
        else:
            return self.level2.time.shape[-1]

    # def to_xarray(self):


def _split_dict_by(dct: dict, keys: list[str]) -> tuple[dict, dict]:
    has_key = {k: v for k, v in dct.items() if k in keys}
    has_key_not = {k: v for k, v in dct.items() if k not in keys}
    return has_key, has_key_not
