from typing import Any, Self
from pydantic import BaseModel, ConfigDict

DefaultFloat = -9999999999999.0
DefaultStr = "undefined"
DefaultInt = -999999999999


class ChannelConfigBaseModel(BaseModel):
    """Base class for channel configurations.

    Fields that are explicitly set (at construction, via assignment or via
    `update`) are tracked by pydantic in `model_fields_set`.
    """

    model_config = ConfigDict(validate_assignment=True, coerce_numbers_to_str=True)

    name: str = ""
    id: int = 0
    type: str = "unknown"
    sign: str = "signed"
    sample_rate: float = 0.0
    units: str = ""

    def __repr__(self) -> str:
        s: str = f"{self.__class__.__name__}:\n"
        for k in self.model_fields_set:
            s += f"    {k}: {getattr(self, k)}\n"
        return s

    def _check_field(self, k: str) -> None:
        if k not in type(self).model_fields:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {k}.")

    def update(self, k: str, v: Any) -> None:
        self._check_field(k)
        setattr(self, k, v)

    def is_set(self, k: str) -> bool:
        self._check_field(k)
        return k in self.model_fields_set

    def clone(self) -> Self:
        return self.model_copy()
