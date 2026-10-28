from typing import Any, Self, TypeVar
from collections.abc import Callable
from pydantic import BaseModel, PrivateAttr

DefaultFloat = -9999999999999.0
DefaultStr = "undefined"
DefaultInt = -999999999999


class ChannelConfigBaseModel(BaseModel):
    name: str = ""
    id: int = 0
    type: str = "unknown"
    sign: str = "signed"
    sample_rate: float = 0.0
    units: str = ""

    _set_properties: list[str] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self._set_properties = list(self.model_fields_set)

    def __repr__(self) -> str:
        s: str = f"{self.__class__.__name__}:\n"
        for k in self._set_properties:
            s += f"    {k}: {getattr(self, k)}\n"
        return s

    def update(self, k: str, v: Any) -> None:
        if not hasattr(self, k):
            raise AttributeError(f"{self.__class__.__name__} has no attribute {k}.")
        setattr(self, k, v)
        if k not in self._set_properties:
            self._set_properties.append(k)

    def is_set(self, k: str) -> bool:
        if not hasattr(self, k):
            raise AttributeError(f"{self.__class__.__name__} has no attribute {k}.")
        return k in self._set_properties

    def clone(self) -> Self:
        new = self.model_copy()
        object.__setattr__(new, "_set_properties", list(self._set_properties))
        return new


# Create a registry of ChannelConfig data classes:

T = TypeVar(
    "T", bound=ChannelConfigBaseModel
)  # A type machting also all its derivates.
_CHANNEL_CONFIG_REGISTRY: dict[str, type[ChannelConfigBaseModel]] = {}


def register_channel_config(instrument_prefix: str,
                            names: list[str]) -> Callable[[type[T]], type[T]]:
    def wrapper(cls: type[T]) -> type[T]:
        for name in names:
            key = f"{instrument_prefix}{name}"
            if key in _CHANNEL_CONFIG_REGISTRY:
                raise ValueError(f"Trying to register a channel config that's already"
                                 f"registered ({_CHANNEL_CONFIG_REGISTRY[key].__name__})"
                                 )
            _CHANNEL_CONFIG_REGISTRY[key] = cls
        return cls

    return wrapper


def channel_config_factory(name: str, prefix: str) -> ChannelConfigBaseModel:
    key = f"{prefix}{name}"
    if key not in _CHANNEL_CONFIG_REGISTRY:
        raise ValueError(f"{key} is not a valid channel name.")
    return _CHANNEL_CONFIG_REGISTRY[key](name=name)
