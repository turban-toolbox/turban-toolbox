"""Defines high-level API to interact with TURBAN toolbox"""

from functools import wraps
from inspect import isclass
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


def get_type_hints_recursive(obj):
    """
    Collect all type definitions on a given objects, also from parent classes.

    :param obj: Any python object
    """
    hints = {}
    # loop through MRO in reverse order if anything is superseded by inheritance
    for type_ in type(obj).mro()[::-1]:
        hints.update(get_type_hints(type_))

    return hints


@dataclass(kw_only=True)
class TimeseriesLevel:
    time: Float[ndarray, "time"]

    _coords: ClassVar[list[str]] = ["time"]
    _level: int

    def arrays_as_xr_dicts(self):
        dct = {
            name: (
                [dim.name for dim in t.dims],
                getattr(self, name),
            )
            for name, t in get_type_hints_recursive(self).items()
            if isclass(t)
            if issubclass(t, AbstractArray)
        }
        data_vars = {k: v for k, v in dct.items() if k not in self._coords}
        coords = {k: v for k, v in dct.items() if k in self._coords}
        print(type(self), data_vars.keys(), coords.keys())

        return data_vars, coords

    def to_xarray(self):
        data_vars, coords = self.arrays_as_xr_dicts()
        # print(data_vars.keys(), coords.keys())
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def get_attr(self, name):
        attr = getattr(self, name)
        if isinstance(attr, SegmentConfig):
            return getattr()


@dataclass(kw_only=True)
class AuxiliaryData(TimeseriesLevel):
    """Mix-in class. Only useful for Level1 and Level2"""

    # Here auxiliary data are stored along with aggregation instructions
    _aux_data: AggAuxDataTypehint | AuxDataTypehint | None = None

    def __post_init__(self):
        if self._aux_data is None:
            self._aux_data = {}

    def arrays_as_xr_dicts(self):
        data_vars, coords = super().arrays_as_xr_dicts()
        if self._level <= 2:
            data = cast(AggAuxDataTypehint, self._aux_data)
            data_vars.update(
                {varname: (dims, arr) for varname, (dims, arr, _) in data.items()}
            )
        else:
            data = cast(AuxDataTypehint, self._aux_data)
            data_vars.update(
                {varname: (dims, arr) for varname, (dims, arr) in data.items()}
            )
        return data_vars, coords

    def _variable_present(self, name):
        """Returns True if variable called `name` already present somewhere"""
        if name in self._aux_data:
            warnings.warn(f"Variable`{name}` already exists in aux data, skipping")
        elif name in get_type_hints_recursive(self):
            warnings.warn(
                f"Variable `{name}` already defined on object itself, skipping"
            )
        else:
            return False

        return True

    def add_aux_data(
        self,
        name: str,
        data: Num[ndarray, "time"] | AggAuxDataTypehint | AuxDataTypehint | None,
        agg_method: str | None = "mean",
        name_out: str | None = None,
    ):
        """Adds auxiliary data to any level.

        If data is a 1D numpy array, uses simplified API. Must supply `agg_method` and optionally `name_out`.

        Otherwise, use the full API
        """
        if data is None:
            self._aux_data = {}

        elif isinstance(data, ndarray):
            # Simplified API
            if agg_method is None:
                raise ValueError("`agg_method` must not be None")
            data = cast(Num[ndarray, "time"], data)
            # so linters understand we have a dict after __post_init__
            if not self._variable_present(name):
                if self._level <= 2:
                    self._aux_data = cast(AggAuxDataTypehint, self._aux_data)
                    agg_method = cast(str, agg_method)  # agg_method must be str
                    # Construct the full aggregation instruction
                    self._aux_data[name] = (["time"], data, {agg_method: name_out})
                else:
                    self._aux_data = cast(AuxDataTypehint, self._aux_data)
                    self._aux_data[name] = (["time"], data)

        else:
            # Full API
            data = cast(AggAuxDataTypehint | AuxDataTypehint, data)
            if self._level <= 2:
                self._aux_data = cast(AggAuxDataTypehint, self._aux_data)
            else:
                self._aux_data = cast(AuxDataTypehint, self._aux_data)
            # clean data for variables that are not present
            data = {
                varname: v
                for varname, v in data.items()
                if not self._variable_present(varname)
            }
            self._aux_data = data


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
    def cfg(self) -> SegmentConfig:
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
class Level1(AuxiliaryData):
    senspeed: Float[ndarray, "time"]
    cfg: SegmentConfig  # only define this here - other levels get it through HasLevelBelow
    section_number: Int[ndarray, "time"]

    _level: int = 1

    # TODO should consider using pydantic or similar for runtime checking of user input
    # (e.g., positive platform speed, etc.)


@dataclass(kw_only=True)
class Level2(HasLevelBelow, AuxiliaryData):
    senspeed: Float[ndarray, "time"]
    section_number: Int[ndarray, "time"]

    _level: int = 2

    @classmethod
    def _from_level_below_kwarg(cls, data: Level1) -> dict:
        return dict(
            time=data.time,
            senspeed=data.senspeed,
            section_number=data.section_number,
            _aux_data=data._aux_data,
        )


@dataclass(kw_only=True)
class Level3(HasLevelBelow, AuxiliaryData):
    waveno: Float[ndarray, "time waveno"]
    freq: Float[ndarray, "waveno"]
    senspeed: Float[ndarray, "time"]
    section_number: Int[ndarray, "time"]

    _coords = ["time", "freq"]

    _level: int = 3

    @classmethod
    def _from_level_below_kwarg(cls, data: Level2) -> dict:
        return dict(
            time=data.time,
            _aux_data=cls.agg(
                data=cast(AggAuxDataTypehint, data._aux_data),
                data_len=len(data.time),
                chunk_length=data.cfg.chunk_length,
                chunk_overlap=data.cfg.chunk_overlap,
                section_number=data.section_number,
            ),
        )

    @classmethod
    def agg(
        cls,
        data: AggAuxDataTypehint,
        data_len: int,
        chunk_length: int,
        chunk_overlap: int,
        section_number: Int[ndarray, "time"],
    ) -> AuxDataTypehint:
        """Aggregates data from Level2. These are stored in  in the form:
        {variable_name_fast: ([dims], ndarray, {agg_method: variable_new_slow}])}
        """
        slow = {}

        # sample_data = data[data.keys()[0]][1]
        cidx = get_chunking_index(
            section_number,
            (chunk_length, chunk_overlap),
            (chunk_length, 0),
        )

        for varname, (dims, arr, rename_dict) in data.items():
            for agg_method, varname_new in rename_dict.items():
                if varname_new is None:
                    varname_new = f"{varname}_{agg_method}"
                slow[varname_new] = (
                    dims,
                    agg_fast_to_slow(
                        arr,
                        reshape_index=cidx,
                        agg_method=agg_method,
                    ),
                )
        return slow


@dataclass(kw_only=True)
class Level4(HasLevelBelow, AuxiliaryData):
    section_number: Int[ndarray, "time"]

    _level: int = 4

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
