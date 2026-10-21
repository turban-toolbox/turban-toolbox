"""Defines high-level API to interact with TURBAN toolbox"""

from enum import IntEnum
from functools import wraps
from inspect import isclass
from typing import get_type_hints, ClassVar, Self, cast
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


class Level(IntEnum):
    """Processing level number (see the four processing levels in the manual).

    Member names are used as node names when exporting to an xarray DataTree."""

    level1 = 1
    level2 = 2
    level3 = 3
    level4 = 4


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
    _level: ClassVar[Level]

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

    def to_xarray(self) -> xr.Dataset:
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
    def from_xarray(cls, ds: xr.Dataset | xr.DataTree) -> Self:
        """Instantiate this level from an xarray Dataset or DataTree.

        Parameters
        ----------
        ds : xr.Dataset or xr.DataTree
            Either a Dataset holding only this level, or a DataTree with nodes
            ``level1`` ... ``level4`` (as written by ``Processing.to_xarray``),
            in which case only the node of this level is parsed (``level_below``
            is not reconstructed).

        Returns
        -------
        TimeseriesLevel
            New instance populated from the dataset.
        """
        if isinstance(ds, xr.DataTree):
            return cls._from_datatree(ds)
        else:
            kwarg = cls._from_xarray_kwarg(ds)
            kwarg["cfg"] = get_type_hints_recursive(cls)["cfg"].from_xarray(ds)
            return cls(**kwarg)

    @classmethod
    def from_netcdf(cls, fname: str) -> Self:
        """Instantiate this level from a netCDF file written by ``Processing.to_netcdf``.

        See ``from_xarray`` for DataTree input.
        """
        return cls.from_xarray(xr.load_datatree(fname))

    @classmethod
    def _from_xarray_kwarg(cls, ds: xr.Dataset) -> dict:
        """Provides constructor kwargs (without `cfg`) parsed from a Dataset."""
        data = {}
        hints = get_type_hints_recursive(cls)
        for name, arr in dict(**ds.data_vars, **ds.coords).items():
            if name in hints:
                logger.debug(f"Adding `{name}` to data")
                data[name] = arr.values
        return data

    @classmethod
    def _from_datatree(cls, tree: xr.DataTree) -> Self:
        """Instantiate this level from its node in a DataTree."""
        return cls.from_xarray(tree[cls._level.name].to_dataset())

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

    @property
    def aux_data(self) -> AuxDataTypehintLevel12 | AuxDataTypehintLevel34:
        """All auxiliary data available on this level.

        For levels taking over auxiliary data from the level below, this is the
        auxiliary data of the level below, updated with those added on this level.
        Entries are references, so data of the level below are not copied.
        """
        below = getattr(self, "level_below", None)
        if self._level in (2, 4) and isinstance(below, AuxiliaryData):
            return cast(dict, below.aux_data) | cast(dict, self._aux_data)
        return cast(dict, self._aux_data)

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
            data = cast(AuxDataTypehintLevel12, self.aux_data)
            data_vars.update(
                {varname: (dims, arr) for varname, (dims, arr, _) in data.items()}
            )
        else:
            data = cast(AuxDataTypehintLevel34, self.aux_data)
            data_vars.update(
                {varname: (dims, arr) for varname, (dims, arr) in data.items()}
            )
        return data_vars, coords

    @classmethod
    def _from_xarray_kwarg(cls, ds: xr.Dataset) -> dict:
        """Provides constructor kwargs, including auxiliary data.

        All data variables not recognised as core fields become auxiliary data.
        """
        kwarg = super()._from_xarray_kwarg(ds)
        kwarg["_aux_data"] = cls._aux_from_xarray(ds)
        return kwarg

    @classmethod
    def _aux_from_xarray(
        cls, ds: xr.Dataset
    ) -> AuxDataTypehintLevel12 | AuxDataTypehintLevel34:
        """Auxiliary data: all data variables not recognised as core fields."""
        hints = get_type_hints_recursive(cls)
        aux = {}
        for name, arr in ds.data_vars.items():
            if name in hints:
                continue
            logger.debug(f"Adding `{name}` to auxiliary data")
            if cls._level <= 2:
                aux[name] = (list(arr.dims), arr.values, {"mean": None})
            else:
                aux[name] = (list(arr.dims), arr.values)
        return aux

    @classmethod
    def _from_datatree(cls, tree: xr.DataTree) -> Self:
        """Instantiate this level from its node in a DataTree.

        Level2/4 additionally take the auxiliary data stored in the level1/3 node
        into their own `_aux_data`, without reconstructing `level_below`.
        """
        new = super()._from_datatree(tree)
        if cls._level in (2, 4):
            below_cls = cast(
                type[AuxiliaryData], cast(type[HasLevelBelow], cls).level_below_cls
            )
            if below_cls._level.name in tree.children:
                aux_below = below_cls._aux_from_xarray(
                    tree[below_cls._level.name].to_dataset()
                )
                new._aux_data = cast(dict, aux_below) | cast(dict, new._aux_data)
        return new

    def _variable_present(self, name):
        """Returns True if variable called `name` already present somewhere"""
        if name in self.aux_data:
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
class HasLevelBelow[L: TimeseriesLevel](TimeseriesLevel, ABC):
    """Base class for levels derived from a level below of type `L`.

    Subclasses specify the level below via the type parameter,
    e.g. `class Level2(HasLevelBelow[Level1], ...)`.
    """

    level_below: L | None = None
    # class of the level below, used when parsing a DataTree
    level_below_cls: ClassVar[type[TimeseriesLevel]]

    @classmethod
    @abstractmethod
    def _from_level_below_kwarg(cls, data: L) -> dict:
        """Provides dictionary `kwarg` that can be passed into class constructor as cls(**kwarg).

        `level_below` and `cfg` need not be included, they are set by `from_level_below`.
        """
        raise NotImplementedError

    @classmethod
    def from_level_below(cls, data: L) -> Self:
        """Construct this level from the level-below data object.

        ``level_below`` is always set to ``data``, and ``cfg`` is shared with it.

        Parameters
        ----------
        data : L
            Data from the level below.

        Returns
        -------
        HasLevelBelow
            New instance initialised from ``data``.
        """
        kwarg = cls._from_level_below_kwarg(data)
        kwarg.update(level_below=data, cfg=data.cfg)
        return cls(**kwarg)

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

    _level: ClassVar[Level] = Level.level1

    # TODO should consider using pydantic or similar for runtime checking of user input
    # (e.g., positive platform speed, etc.)


@dataclass(kw_only=True)
class Level2[L: Level1](HasLevelBelow[L], AuxiliaryData):
    senspeed: Float[ndarray, "time"]
    section_number: Int[ndarray, "time"]

    _level: ClassVar[Level] = Level.level2
    level_below_cls: ClassVar[type[TimeseriesLevel]] = Level1

    @classmethod
    def _from_level_below_kwarg(cls, data: L) -> dict:
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
            # aux data of the level below are read through `aux_data`, only
            # aux data added on this level are stored here
            _aux_data={},
        )


@dataclass(kw_only=True)
class Level3[L: Level2](HasLevelBelow[L], AuxiliaryData):
    waveno: Float[ndarray, "time waveno"]
    freq: Float[ndarray, "waveno"]
    senspeed: Float[ndarray, "time"]
    section_number: Int[ndarray, "time"]

    _coords = ["time", "freq"]

    _level: ClassVar[Level] = Level.level3
    level_below_cls: ClassVar[type[TimeseriesLevel]] = Level2

    @classmethod
    def _from_level_below_kwarg(cls, data: L) -> dict:
        """Build constructor kwargs for Level3 from Level2 data, aggregating aux data.

        Fast variables are aggregated to chunks; `freq` is the frequency vector of
        the FFT segments and `waveno` is obtained from it via the chunk-mean `senspeed`.

        Parameters
        ----------
        data : Level2
            Level 2 data object.

        Returns
        -------
        dict
            Keyword arguments to pass to the Level3 constructor.
        """
        cfg = data.cfg
        ii = get_chunking_index(
            data.section_number, (cfg.chunk_length, cfg.chunk_overlap)
        )
        senspeed = agg_fast_to_slow(data.senspeed, reshape_index=ii)
        freq = np.fft.rfftfreq(cfg.segment_length, d=1 / cfg.sampfreq)
        return dict(
            time=agg_fast_to_slow(data.time, reshape_index=ii, agg_method="take_mid"),
            freq=freq,
            waveno=freq[newaxis, :] / senspeed[:, newaxis],
            senspeed=senspeed,
            section_number=agg_fast_to_slow(
                data.section_number, reshape_index=ii, agg_method="max"
            ),
            _aux_data=cls.agg(
                data=cast(AuxDataTypehintLevel12, data.aux_data),
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
class Level4[L: Level3](HasLevelBelow[L], AuxiliaryData):
    section_number: Int[ndarray, "time"]

    _level: ClassVar[Level] = Level.level4
    level_below_cls: ClassVar[type[TimeseriesLevel]] = Level3

    @classmethod
    def _from_level_below_kwarg(cls, data: L) -> dict:
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
            # aux data of the level below are read through `aux_data`, only
            # aux data added on this level are stored here
            _aux_data={},
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


class Processing:
    """Propagates a given start level up to level 4.
    
    Every subclass must define its own ``_level_mapping``."""

    # Mapping from level number to the corresponding level class
    _level_mapping: ClassVar[dict[Level, type[TimeseriesLevel]]] = {
        Level.level1: Level1,
        Level.level2: Level2,
        Level.level3: Level3,
        Level.level4: Level4,
    }

    def __init_subclass__(cls, **kwargs):
        """Require every subclass to define its own ``_level_mapping``."""
        super().__init_subclass__(**kwargs)
        if "_level_mapping" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define its own `_level_mapping`")

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
        for l in Level:
            if l > data._level:
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

    def to_xarray(self) -> xr.DataTree:
        """Export all available levels to a DataTree with nodes ``level1`` ... ``level4``.

        Auxiliary data shared with the level below (Level2 from Level1, Level4
        from Level3) are only stored on the lower level.
        """
        # level whose auxiliary data are not stored on the given level
        aux_source = {Level.level2: self.level1, Level.level4: self.level3}
        out_data = {}
        for data in [self.level1, self.level2, self.level3, self.level4]:
            if data is not None:
                out = data.to_xarray()
                source = aux_source.get(data._level)
                if source is not None:
                    out = out.drop_vars(list(source.aux_data), errors="ignore")
                out_data[data._level.name] = xr.DataTree(out)

        data_tree = xr.DataTree(children=out_data)
        return data_tree

    @classmethod
    def from_xarray(cls, data_tree: xr.DataTree) -> Self:
        """Instantiate pipeline from a DataTree.

        Every available level is parsed from its own node and linked to the level
        below, so that Level2/4 read auxiliary data through Level1/3.
        """
        data = None
        for level, class_ in cls._level_mapping.items():
            if level.name in data_tree.children:
                new = class_.from_xarray(data_tree[level.name].to_dataset())
                if data is not None:
                    cast(HasLevelBelow, new).level_below = data
                data = new
        if data is None:
            raise ValueError(f"Could not find any data.")

        return cls(data)

    def to_netcdf(self, fname: str):
        """Write all available levels to a netCDF file, see ``to_xarray``."""
        self.to_xarray().to_netcdf(fname)

    @classmethod
    def from_netcdf(cls, fname: str) -> Self:
        """Instantiate pipeline from a netCDF file, see ``from_xarray``."""
        return cls.from_xarray(xr.load_datatree(fname))


# def _split_dict_by(dct: dict, keys: list[str]) -> tuple[dict, dict]:
#     has_key = {k: v for k, v in dct.items() if k in keys}
#     has_key_not = {k: v for k, v in dct.items() if k not in keys}
#     return has_key, has_key_not
