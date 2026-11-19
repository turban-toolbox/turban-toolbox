from typing import Literal

from pydantic import BaseModel, field_validator
from enum import Enum

from turban.process.generic.config import SegmentConfig

type SpectrumModel = Literal["batchelor", "kraichnan"]
# keys of FP07_RESPONSE_MODELS, or "none" for no response correction
type FP07ResponseModelName = Literal[
    "nash99", "rsi", "gm", "kocsis", "nash02", "sommer"
]


class FP07Response(BaseModel):
    # tau = tau0 * (W / 1 m s-1) ** exponent
    name: FP07ResponseModelName
    tau0: float  # response time [s]
    gamma: float  # exponent of speed,
    npoles: int  # number of poles


FP07_RESPONSE_MODELS: dict[FP07ResponseModelName, FP07Response] = {
    # Nash et al. (1999), as in Piccolroaz et al. (2021)
    "nash99": FP07Response(name="nash99", tau0=7e-3, gamma=-0.12, npoles=1),
    # Vachon & Lueck (1984), RSI default (F0 = 25 W^0.5)
    "rsi": FP07Response(name="rsi", tau0=4.1e-3, gamma=-0.5, npoles=2),
    # Gregg & Meagher (1980)
    "gm": FP07Response(name="gm", tau0=10.6e-3, gamma=-0.32, npoles=2),
    # Kocsis et al. (1999)
    "kocsis": FP07Response(name="kocsis", tau0=7e-3, gamma=-0.32, npoles=2),
    # Nash et al. (2002) according to Sommer et al. (2013)
    "nash02": FP07Response(name="nash02", tau0=8.5e-3, gamma=0.0, npoles=2),
    # Sommer et al. (2013)
    "sommer": FP07Response(name="sommer", tau0=10e-3, gamma=0.0, npoles=2),
}


class UTempConfig(SegmentConfig):
    """Configures processing of timeseries using temperature gradient spectra.

    OBS: fp07_response can also be instantiated as string (one of keys of
    FP07_RESPONSE_MODELS)
    """

    # --- Level 3: frequency response of the FP07 thermistor
    fp07_response: FP07Response | None = FP07_RESPONSE_MODELS["nash99"]

    # --- Level 4: physical properties (if no auxiliary data `temp`)
    molvisc_fallback: float = 1.6e-6  # [m2 s-1]
    moldiff_fallback: float = 1.4e-7  # [m2 s-1]

    # --- Level 4: spectral fitting
    spectrum_model: SpectrumModel = "batchelor"
    q_batchelor: float = 3.7  # Oakey (1982)
    q_kraichnan: float = 5.26  # Bogucki et al. (1997)
    waveno_limit_upper: float | None = None  # [cpm] optional hard upper limit

    @field_validator("fp07_response", mode="before")
    @classmethod
    def _resolve_fp07_response(cls, v):
        if isinstance(v, str):
            try:
                return FP07_RESPONSE_MODELS[v]
            except KeyError:
                raise ValueError(
                    f"Unknown FP07 response model {v!r}. "
                    f"Choose one of {sorted(FP07_RESPONSE_MODELS)}, "
                    "pass an FP07Response compliant dict or similar,"
                    "or None."
                ) from None
        return v

    @property
    def dof_spectrum(self) -> float:
        """Degrees of freedom of the spectral estimates, 1.9 N_f (Nuttall 1971)."""
        return 1.9 * self.number_fft_windows_per_chunk

    @property
    def q(self) -> float:
        return {"batchelor": self.q_batchelor, "kraichnan": self.q_kraichnan}[
            self.spectrum_model
        ]


#########################
