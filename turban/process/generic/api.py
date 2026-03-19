"""Defines high-level API to interact with TURBAN toolbox"""

from functools import wraps
from inspect import isclass
from typing import get_type_hints, ClassVar, cast
from abc import abstractmethod, ABC
from typing import Literal
from dataclasses import dataclass
from jaxtyping import Float, Int, AbstractArray, Num, Shaped
from turban.process.generic.config import SegmentConfig
from numpy import newaxis, nan, ndarray
import numpy as np
import xarray as xr

from turban.utils.util import agg_fast_to_slow, get_chunking_index
from turban.variables import VARIABLES
from turban.utils.logging import get_logger

logger = get_logger(__name__)

# For Level1/2
AuxDataTypehintLevel12 = dict[
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
AuxDataTypehintLevel34 = dict[
    str,  # variable name
    tuple[
        list[str],  # list of dimensions (for xarray)
        Num[ndarray, "*any time_fast"],  # data
    ],
]


def get_type_hints_recursive(cls) -> dict:
    """
    Collect all type definitions on a given class, also from parent classes.

    :param obj: Any python object
    """
    hints = {}
    # loop through MRO in reverse order if anything is superseded by inheritance
    for type_ in cls.mro()[::-1]:
        hints.update(get_type_hints(type_))

    return hints


@dataclass(kw_only=True)
class TimeseriesLevel:
    time: Shaped[ndarray, "time"]

    cfg: SegmentConfig  # only define this here - other levels get it through HasLevelBelow
    _coords: ClassVar[list[str]] = ["time"]
    _level: ClassVar[int]

    def arrays_as_xr_dicts(self):
        """Separate array-typed fields into data-variable and coordinate dicts.

        Returns
        -------
        data_vars : dict
            Fields not listed in ``_coords``, in xarray ``(dims, data)`` format.
        coords : dict
            Fields listed in ``_coords``, in xarray ``(dims, data)`` format.
        """
        dct = {
            name: (
                [dim.name for dim in t.dims],
                getattr(self, name),
            )
            for name, t in get_type_hints_recursive(type(self)).items()
            if isclass(t)
            if issubclass(t, AbstractArray)
        }
        data_vars = {k: v for k, v in dct.items() if k not in self._coords}
        coords = {k: v for k, v in dct.items() if k in self._coords}
        return data_vars, coords

    def to_xarray(self):
        """Export this level to an xarray Dataset.

        Returns
        -------
        xr.Dataset
            Dataset containing all array fields as data variables and coordinates,
            with TURBAN standard attributes and config stored as global attributes.
        """
        data_vars, coords = self.arrays_as_xr_dicts()
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        for varname in set(list(ds.data_vars) + list(ds.coords)):
            ds[varname].attrs.update(VARIABLES.get(varname, {}))

        # add config as global attributes
        self.cfg.add_to_xarray(ds)
        return ds

    @classmethod
    def from_xarray(cls, ds: xr.Dataset):
        """Instantiate this level from an xarray Dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing array fields and config global attributes.

        Returns
        -------
        TimeseriesLevel
            New instance populated from the dataset.
        """
        data = {}
        hints = get_type_hints_recursive(cls)
        for name, arr in dict(**ds.data_vars, **ds.coords).items():
            if name in hints:
                logger.debug(f"Adding `{name}` to data")
                data[name] = arr.values

        data["cfg"] = hints["cfg"].from_xarray(ds)

        return cls(**data)

    def get_attr(self, name):
        """Retrieve an attribute by name.

        Parameters
        ----------
        name : str
            Name of the attribute to retrieve.

        Returns
        -------
        object
            Value of the attribute.
        """
        attr = getattr(self, name)
        if isinstance(attr, SegmentConfig):
            return getattr()


@dataclass(kw_only=True)
class AuxiliaryData(TimeseriesLevel):
    """Mix-in class to provide support for non-essential auxiliary data"""

    # Here auxiliary data are stored along with aggregation instructions
    _aux_data: AuxDataTypehintLevel12 | AuxDataTypehintLevel34 | None = None

    def __post_init__(self):
        """Initialise ``_aux_data`` to an empty dict if it was not provided."""
        if self._aux_data is None:
            self._aux_data = {}

    def arrays_as_xr_dicts(self):
        """Separate array fields including auxiliary data into data-variable and coordinate dicts.

        Returns
        -------
        data_vars : dict
            Fields (including auxiliary data) not listed in ``_coords``.
        coords : dict
            Fields listed in ``_coords``.
        """
        data_vars, coords = super().arrays_as_xr_dicts()
        if self._level <= 2:
            data = cast(AuxDataTypehintLevel12, self._aux_data)
            data_vars.update(
                {varname: (dims, arr) for varname, (dims, arr, _) in data.items()}
            )
        else:
            data = cast(AuxDataTypehintLevel34, self._aux_data)
            data_vars.update(
                {varname: (dims, arr) for varname, (dims, arr) in data.items()}
            )
        return data_vars, coords

    @classmethod
    def from_xarray(cls, ds: xr.Dataset):
        """Instantiate from an xarray Dataset, loading auxiliary variables.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset; variables not recognised as core fields are added as auxiliary data.

        Returns
        -------
        AuxiliaryData
            New instance with core and auxiliary data populated.
        """
        new = super().from_xarray(ds)
        for name in ds:
            if not hasattr(new, name):
                if len(ds[name].shape) == 1:
                    new.add_aux_data(ds[name].values, name=name)
                else:
                    logger.warning(
                        (
                            f"Skipping potential auxiliary variable `{name}` defined on "
                            f"dataset as it has {len(ds[name].shape)} dimensions"
                        )
                    )
        return new

    def _variable_present(self, name):
        """Returns True if variable called `name` already present somewhere"""
        if name in self._aux_data:
            logger.warning(f"Variable`{name}` already exists in aux data, skipping")
        elif name in get_type_hints_recursive(type(self)):
            logger.warning(
                f"Variable `{name}` already defined on object itself, skipping"
            )
        else:
            return False

        return True

    def add_aux_data(
        self,
        data: (
            Num[ndarray, "time"]
            | AuxDataTypehintLevel12
            | AuxDataTypehintLevel34
            | None
        ),
        name: str | None = None,
        agg_method: str | None = "mean",
        name_out: str | None = None,
    ):
        """Adds auxiliary data to any level.

        If data is a 1D numpy array, uses simplified API.
        Must then supply `name`, `agg_method` and optionally `name_out`.

        Otherwise, use the full API.
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
                    self._aux_data = cast(AuxDataTypehintLevel12, self._aux_data)
                    agg_method = cast(str, agg_method)  # agg_method must be str
                    # Construct the full aggregation instruction
                    self._aux_data[name] = (["time"], data, {agg_method: name_out})
                else:
                    self._aux_data = cast(AuxDataTypehintLevel34, self._aux_data)
                    self._aux_data[name] = (["time"], data)

        else:
            # Full API
            data = cast(AuxDataTypehintLevel12 | AuxDataTypehintLevel34, data)
            if self._level <= 2:
                self._aux_data = cast(AuxDataTypehintLevel12, self._aux_data)
            else:
                self._aux_data = cast(AuxDataTypehintLevel34, self._aux_data)
            # clean data for variables that are not present
            data = {
                varname: v
                for varname, v in data.items()
                if not self._variable_present(varname)
            }
            self._aux_data = data


@dataclass(kw_only=True)
class HasLevelBelow(TimeseriesLevel):
    level_below: TimeseriesLevel | None = None

    @classmethod
    @abstractmethod
    def _from_level_below_kwarg(cls, data: TimeseriesLevel | None) -> dict:
        """Provides dictionary `kwarg` that can be passed into class constructor as cls(**kwarg)."""
        return {}

    @classmethod
    def from_level_below(cls, data: TimeseriesLevel | None):
        """Construct this level from the level-below data object.

        Parameters
        ----------
        data : TimeseriesLevel or None
            Data from the level below.

        Returns
        -------
        HasLevelBelow
            New instance initialised from ``data``.
        """
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
    section_number: Int[ndarray, "time"]

    _level: ClassVar[int] = 1

    # TODO should consider using pydantic or similar for runtime checking of user input
    # (e.g., positive platform speed, etc.)


@dataclass(kw_only=True)
class Level2(HasLevelBelow, AuxiliaryData):
    senspeed: Float[ndarray, "time"]
    section_number: Int[ndarray, "time"]

    _level: ClassVar[int] = 2

    @classmethod
    def _from_level_below_kwarg(cls, data: Level1) -> dict:
        """Build constructor kwargs for Level2 from Level1 data.

        Parameters
        ----------
        data : Level1
            Level 1 data object.

        Returns
        -------
        dict
            Keyword arguments to pass to the Level2 constructor.
        """
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

    _level: ClassVar[int] = 3

    @classmethod
    def _from_level_below_kwarg(cls, data: Level2) -> dict:
        """Build constructor kwargs for Level3 from Level2 data, aggregating aux data.

        Parameters
        ----------
        data : Level2
            Level 2 data object.

        Returns
        -------
        dict
            Keyword arguments to pass to the Level3 constructor.
        """
        return dict(
            time=data.time,
            _aux_data=cls.agg(
                data=cast(AuxDataTypehintLevel12, data._aux_data),
                data_len=len(data.time),
                chunk_length=data.cfg.chunk_length,
                chunk_overlap=data.cfg.chunk_overlap,
                section_number=data.section_number,
            ),
        )

    @classmethod
    def agg(
        cls,
        data: AuxDataTypehintLevel12,
        data_len: int,
        chunk_length: int,
        chunk_overlap: int,
        section_number: Int[ndarray, "time"],
    ) -> AuxDataTypehintLevel34:
        """Aggregates data from Level2. These are stored in  in the form:
        {variable_name_fast: ([dims], ndarray, {agg_method: variable_new_slow}])}
        """
        slow = {}

        for varname, (dims, arr, rename_dict) in data.items():
            for agg_method, varname_new in rename_dict.items():
                if varname_new is None:
                    varname_new = f"{varname}_{agg_method}"
                slow[varname_new] = (
                    dims,
                    agg_fast_to_slow(
                        arr,
                        section_number_or_data_len=section_number,
                        chunk_length=chunk_length,
                        chunk_overlap=chunk_overlap,
                        agg_method=agg_method,
                    ),
                )
        return slow


@dataclass(kw_only=True)
class Level4(HasLevelBelow, AuxiliaryData):
    section_number: Int[ndarray, "time"]

    _level: ClassVar[int] = 4

    @classmethod
    def _from_level_below_kwarg(cls, data: Level3) -> dict:
        """Build constructor kwargs for Level4 from Level3 data.

        Parameters
        ----------
        data : Level3
            Level 3 data object.

        Returns
        -------
        dict
            Keyword arguments to pass to the Level4 constructor.
        """
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
        """Mapping from level number to the corresponding level class.

        Returns
        -------
        dict
            Dict of ``{int: TimeseriesLevel subclass}`` for levels 1–4.
        """
        # Slightly clumsy way of requiring a class attribute called _level_mapping
        return {1: Level1, 2: Level2, 3: Level3, 4: Level4}

    def __init__(
        self,
        data: TimeseriesLevel,
    ):
        """Propagate ``data`` up to level 4 using the level mapping.

        Parameters
        ----------
        data : TimeseriesLevel
            Starting data object at any level.
        """
        for l in range(data._level + 1, 5):
            data = self._level_mapping[l].from_level_below(data)
        self.data = data

    @property
    def level1(self):
        """Level 1 data, or None if not available."""
        if self.level2 is None:
            return None
        else:
            return self.level2.level_below

    @property
    def level2(self):
        """Level 2 data, or None if not available."""
        if self.level3 is None:
            return None
        else:
            return self.level3.level_below

    @property
    def level3(self):
        """Level 3 data, or None if not available."""
        if self.level4 is None:
            return None
        else:
            return self.level4.level_below

    @property
    def level4(self):
        """Level 4 data (the topmost processed level)."""
        return self.data

    @property
    def cfg(self):
        """Segment configuration from the top-level data object."""
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
        out_data = {}
        for data in [self.level1, self.level2, self.level3, self.level4]:
            if data is not None:
                out = data.to_xarray()
                out_data[f"level{data._level}"] = xr.DataTree(out)

        data_tree = xr.DataTree(children=out_data)
        return data_tree

    @classmethod
    def from_xarray(cls, data_tree: xr.DataTree):
        """Instantiate pipeline from lowest available level"""
        for level, class_ in cls._level_mapping.items():
            level_str = f"level{level:d}"
            if level_str in data_tree:
                logger.debug(f"Start processing from level {level}")
                data = class_.from_xarray(data_tree[level_str].to_dataset())
                break
        else:
            raise ValueError(f"Could not find any data.")

        return cls(data)


# def _split_dict_by(dct: dict, keys: list[str]) -> tuple[dict, dict]:
#     has_key = {k: v for k, v in dct.items() if k in keys}
#     has_key_not = {k: v for k, v in dct.items() if k not in keys}
#     return has_key, has_key_not
