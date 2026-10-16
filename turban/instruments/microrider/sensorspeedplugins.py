from jaxtyping import Float

import numpy as np
from scipy.interpolate import make_interp_spline

from turban.instruments.microrider import rsIO


def senspeed_constant(
    time: Float[np.ndarray, "time"], speed: float
) -> Float[np.ndarray, "time"]:
    """Return a constant sensor speed for all time steps.

    Parameters
    ----------
    time : Float[np.ndarray, "time"]
        Time vector; used only to determine the output shape.
    speed : float
        Sensor speed in m/s.

    Returns
    -------
    Float[np.ndarray, "time"]
        Sensor speed in m/s, same length as `time`.
    """
    return np.full_like(time, speed)


def senspeed_from_lookup_table(
    time: Float[np.ndarray, "time"],
    time_table: Float[np.ndarray, "n"],
    senspeed_table: Float[np.ndarray, "n"],
) -> Float[np.ndarray, "time"]:
    """Linearly interpolate sensor speed from a lookup table.

    Parameters
    ----------
    time : Float[np.ndarray, "time"]
        Time vector at which sensor speed is evaluated.
    time_table : Float[np.ndarray, "n"]
        Time vector of the lookup table, same time base as `time`.
    senspeed_table : Float[np.ndarray, "n"]
        Sensor speed in m/s at `time_table`.

    Returns
    -------
    Float[np.ndarray, "time"]
        Sensor speed in m/s at `time`.
    """
    return make_interp_spline(time_table, senspeed_table, k=1)(time)


def senspeed_from_em(
    time: Float[np.ndarray, "time"], microrider_data: rsIO.MicroRiderData
) -> Float[np.ndarray, "time"]:
    """Sensor speed from the MicroRider's electromagnetic current meter (``U_EM``).

    The slow-rate ``U_EM`` channel is linearly interpolated onto `time`.

    Parameters
    ----------
    time : Float[np.ndarray, "time"]
        Time vector (seconds since epoch) at which sensor speed is evaluated.
    microrider_data : rsIO.MicroRiderData
        Data read from a .p file, containing a ``U_EM`` channel.

    Returns
    -------
    Float[np.ndarray, "time"]
        Sensor speed in m/s at `time`.
    """
    t_slow = microrider_data.header.t_slow + microrider_data.header.timestamp
    return senspeed_from_lookup_table(time, t_slow, microrider_data.U_EM.data)
