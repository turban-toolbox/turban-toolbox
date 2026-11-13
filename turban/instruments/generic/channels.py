from pydantic import BaseModel, ConfigDict


class ChannelConfigBaseModel(BaseModel):
    """Base class for channel configurations.

    Subclasses declare calibration parameters as required fields, or with a
    real default.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        coerce_numbers_to_str=True,
        extra="forbid",  # reject misspelt/unknown calibration keys
    )

    name: str
    id: int
    type: str = "unknown"
    sign: str = "signed"
    sample_rate: float = 0.0
    units: str = ""
