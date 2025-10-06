"""Defines high-level API to interact with TURBAN toolbox"""

from functools import wraps
from logging import warnings
from typing import get_type_hints, ClassVar, cast
from abc import abstractmethod, ABC
from typing import Literal
from dataclasses import dataclass
from jaxtyping import Float, Int, AbstractArray, Num
from turban.process.generic.config import SegmentConfig
from numpy import newaxis, nan, ndarray
import numpy as np
import xarray as xr

from turban.utils.util import agg_fast_to_slow, get_chunking_index

# For Level1/2
AggAuxDataTypehint = dict[
    str,  # variable name
    tuple[
        list[str],  # list of dimensions (for xarray)
        Num[ndarray, "*any time_fast"],  # data
        dict[
            str,  # aggregation method (e.g. `mean`)
            str | None,  # new variable name
        ],
    ],
]

# For Level3/4
AuxDataTypehint = dict[
    str,  # variable name
    tuple[
        list[str],  # list of dimensions (for xarray)
        Num[ndarray, "*any time_fast"],  # data
    ],
]


@dataclass(kw_only=True)
class TimeseriesLevel:
    time: Float[ndarray, "time"]

    _coords: ClassVar[list[str]] = ["time"]

    def arrays_as_xr_dicts(self):
        dct = {
            name: (
                [dim.name for dim in t.dims],
                getattr(self, name),
            )
            for name, t in get_type_hints(self).items()
            if issubclass(t, AbstractArray)
        }
        data_vars = {k: v for k, v in dct.items() if k not in self._coords}
        coords = {k: v for k, v in dct.items() if k in self._coords}
        return data_vars, coords

    def to_xarray(self):
        data_vars, coords = self.arrays_as_xr_dicts()
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def get_attr(self, name):
        attr = getattr(self, name)
        if isinstance(attr, SegmentConfig):
            return getattr()


@dataclass(kw_only=True)
class AuxiliaryDataFast(TimeseriesLevel):
    """Mix-in class. Only useful for Level1 and Level2"""

    # Here auxiliary data are stored along with aggregation instructions
    _agg_aux_data: AggAuxDataTypehint | None = None

    def __post_init__(self):
        if self._agg_aux_data is None:
            self._agg_aux_data = {}

    def arrays_as_xr_dicts(self):
        data_vars, coords = super().arrays_as_xr_dicts()
        # so linters understand we have a dict after __post_init__
        data = cast(AggAuxDataTypehint, self._agg_aux_data)
        data_vars.update(
            {varname: (dims, arr) for varname, (dims, arr, _) in data.items()}
        )
        return data_vars, coords

    def add_aux_data(
        self,
        name: str,
        data: Float[ndarray, "time"],
        agg_method: str = "mean",
        name_out: str | None = None,
    ):
        """Simplified API"""
        # so linters understand we have a dict after __post_init__
        self._agg_aux_data = cast(AggAuxDataTypehint, self._agg_aux_data)
        if name in self._agg_aux_data:
            raise ValueError(f"Aux data `{name}` already exists")
        # Construct the full aggregation instruction
        self._agg_aux_data[name] = (["time"], data, {agg_method: name_out})


@dataclass(kw_only=True)
class AuxiliaryDataSlow(TimeseriesLevel):
    """Mix-in class. Only useful for Level3 and Level4"""

    _aux_data: AuxDataTypehint | None = None

    def __post_init__(self):
        if self._aux_data is None:
            self._aux_data = {}

    def arrays_as_xr_dicts(self):
        data_vars, coords = super().arrays_as_xr_dicts()
        # so linters understand we have a dict after __post_init__
        data = cast(AuxDataTypehint, self._aux_data)
        data_vars.update(data)
        return data_vars, coords

    def add_aux_data(
        self,
        name: str,
        data: Float[ndarray, "time"],
    ):
        """This simply adds entries"""
        # so linters understand we have a dict after __post_init__
        self._aux_data = cast(AuxDataTypehint, self._aux_data)
        if name in self._aux_data:
            raise ValueError(f"Aux data `{name}` already exists")
        self._aux_data[name] = (["time"], data)


@dataclass(kw_only=True)
class HasLevelBelow(TimeseriesLevel):
    level_below: TimeseriesLevel | None

    @classmethod
    @abstractmethod
    def _from_level_below_kwarg(cls, data: TimeseriesLevel | None) -> dict:
        """Provides dictionary `kwarg` that can be passed into class constructor as cls(**kwarg)."""
        return {}

    @classmethod
    def from_level_below(cls, data: TimeseriesLevel | None):
        return cls(**cls._from_level_below_kwarg(data))

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
class Level1(AuxiliaryDataFast):
    pspd: Float[ndarray, "time"]
    cfg: SegmentConfig  # only define this here - other levels get it through HasLevelBelow
    section_number: Int[ndarray, "time"]


@dataclass(kw_only=True)
class Level2(HasLevelBelow, AuxiliaryDataFast):
    pspd: Float[ndarray, "time"]
    section_number: Int[ndarray, "time"]

    @classmethod
    def _from_level_below_kwarg(cls, data: Level1) -> dict:
        return dict(
            time=data.time,
            pspd=data.pspd,
            section_number=data.section_number,
            _agg_aux_data=data._agg_aux_data,
        )


@dataclass(kw_only=True)
class Level3(HasLevelBelow, AuxiliaryDataSlow):
    waveno: Float[ndarray, "time waveno"]
    freq: Float[ndarray, "waveno"]
    platform_speed: Float[ndarray, "time"]
    section_number: Int[ndarray, "time"]

    _coords = ["time", "freq"]

    @classmethod
    def _from_level_below_kwarg(cls, data: Level2) -> dict:
        return dict(
            time=data.time,
            _aux_data=cls.agg(
                data=cast(AggAuxDataTypehint, data._agg_aux_data),
                data_len=len(data.time),
                diss_length=data.cfg.diss_length,
                diss_overlap=data.cfg.diss_overlap,
                section_number=data.section_number,
            ),
        )

    @classmethod
    def agg(
        cls,
        data: AggAuxDataTypehint,
        data_len: int,
        diss_length: int,
        diss_overlap: int,
        section_number: Int[ndarray, "time"],
    ) -> AuxDataTypehint:
        """Aggregates data from Level2 in the form:
        {variable_name_fast: ([dims], ndarray, {agg_method: variable_new_slow}])}
        """
        slow = {}

        # sample_data = data[data.keys()[0]][1]
        cidx = get_chunking_index(
            data_len,
            diss_length,
            0,
            diss_length,
            diss_overlap,
            section_number,
        )

        for varname, (dims, arr, rename_dict) in data.items():
            for agg_method, varname_new in rename_dict.items():
                if varname_new is None:
                    varname_new = f"{varname}_{agg_method}"
                slow[varname_new] = (
                    dims,
                    agg_fast_to_slow(arr, reshape_index=cidx, agg_method=agg_method),
                )
        return slow


@dataclass(kw_only=True)
class Level4(HasLevelBelow, AuxiliaryDataSlow):
    section_number: Int[ndarray, "time"]

    @classmethod
    def _from_level_below_kwarg(cls, data: Level3) -> dict:
        return dict(
            time=data.time,
            section_number=data.section_number,
            _aux_data=data._aux_data,
        )


# class AggAux:
#     """Aggregates auxiliary variables from fast to slow timesteps"""

#     def __init__(
#         self,
#         data_len: int,
#         diss_length: int,
#         diss_overlap: int,
#         section_number: Int[ndarray, "... data_len"] | None = None,
#     ) -> None:
#         if section_number is None:
#             self.section_number = np.ones((data_len), dtype=int)
#         else:
#             self.section_number = section_number

#         self._data_len = data_len
#         self._diss_length = diss_length
#         self._diss_overlap = diss_overlap
#         # self._agg_index =

#     def fast_to_xarray(self):
#         coords, data_vars = _split_dict_by(self._fast, self._coords)
#         fast = xr.Dataset(data_vars=data_vars, coords=coords)
#         return fast


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
    ):
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

    @property
    def data_len_fast(self):
        """Length of the fast time vector (levels 1 and 2)"""
        if self.level2 is None:
            return None
        else:
            return self.level2.time.shape[-1]

    def to_xarray(self):
        """Export"""
        out_data = []
        for data in [self.level1, self.level2, self.level3, self.level4]:
            if data is None:
                out = None
            else:
                out = data.to_xarray()
            out_data.append(out)
        return out_data


# def _split_dict_by(dct: dict, keys: list[str]) -> tuple[dict, dict]:
#     has_key = {k: v for k, v in dct.items() if k in keys}
#     has_key_not = {k: v for k, v in dct.items() if k not in keys}
#     return has_key, has_key_not
