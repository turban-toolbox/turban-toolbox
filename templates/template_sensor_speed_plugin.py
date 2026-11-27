"""Template code for writing sensor speed plugins.

* Copy this code to a file in your working or processing directory and give it a descriptive name

* Edit the file and
    - rename the SensorSpeedTemplate class to something meaningfull
    - implement the __init__, get_sensor_speed methods and any other methods or functions
      that are required. Note that microstructure raw data are accessible from
      self.microstructure_raw_data (dictionary).

* Your processing script can access the plugin by defining an entry in the MicroriderConfig by
  setting the sensor_speed_plugin keyword as
    sensor_speed_plugin = "<module_filename>:<class_name>"
    and supply any parameters to the constructor as a dictionary according to
    sensor_speed_plugin_parameters: dict[str, float | str] = {}


"""

import numpy as np
from jaxtyping import Float # required for type hinting

import turban.instruments.microrider.sensorspeedplugins as plugins

@plugins.register_plugin
class SensorSpeedTemplate(plugins.SensorSpeedABC):  # <- Rename

    def __init__(
        self,
        # add constructor parameters as needed
    ) -> None:
        """
        Parameters
        ----------
        ...
        """
        super().__init__()

    def get_sensor_speed(
        self, t: Float[np.ndarray, "time"]
    ) -> Float[np.ndarray, "time"]:
        """Return the flight-model speed at the requested times.

        Parameters
        ----------
        t : Float[np.ndarray, "time"]
            Time vector at which sensor speed is evaluated.

        Returns
        -------
        Float[np.ndarray, "time"]
            Speed past sensors in m/s
        """
        # implement this method. Microrider data are accessible through self.microstructure_raw_data
        U = t* 0 # Update this as appropriate. Ensure return value has same dimensions as the time input vector.
        return U
