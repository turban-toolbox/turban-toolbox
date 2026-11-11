import re
from typing import Any


class MicroRiderConfig:
    """Class to handle MicroRider configuration from the raw setupstr
    read from the header information.

    Parameters
    ----------

    setupstring : str
        string as found in the header of the data file, containing there
        configuration of the MicroRider.

    The class provieds methods to query the configuration of a specific
    channel, as well as extracting the raw channel names, cruise and
    instrument information.

    Example
    -------

    >>> cnf = MicroRiderConfig()
    >>> cnf.parse(setupstring)
    ### get the cruise information
    >>> cnf.get_config("cruise_info")
    ### get the config of a given section...
    >>> cnf.get_config("channel08")
    ### get the config of given channel name...
    >>> cnf.get_channel_config("sh1")
    """

    def __init__(self) -> None:
        self._config: dict[str, dict[str, Any]] = {}
        self._current_section: str = "root"
        self._number_of_channels: int = 0

    def parse(self, config_text: str) -> None:
        """
        Parse the configuration text into a nested dictionary.

        Parameters
        ----------
        config_text: str
            Configuration file contents as a string
        """
        # type hint parsed_value
        parsed_value: Any

        channel_index = 0
        # Initialize root section
        self._config["root"] = {}

        # Split the text into lines and process each line
        for line in config_text.split("\n"):
            # Remove leading/trailing whitespace
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith(";"):
                continue
            # replace Any tabs with spaces.
            line = line.replace("\t", " ")
            # Check for section header
            section_match = re.match(r"^\[(\w+)\]$", line)
            if section_match:
                section_name = section_match.group(1)
                # add an index number to all channel sections, as they
                # appear multiples times.
                if section_name == "channel":
                    self._current_section = f"channel{channel_index:02d}"
                    channel_index += 1
                    self._number_of_channels = channel_index
                else:
                    self._current_section = section_name
                self._config[self._current_section] = {}
                continue

            # Parse key-value pairs
            kv_match = re.match(r"(^.+?)\s*;.*$", line)
            if kv_match:  # trailing comment
                line = kv_match.group(1)  # discards comment and Any leading spaces.
            kv_match = re.match(r"^(\w+)\s*=\s*(.+)$", line)
            if kv_match:
                key, value = kv_match.groups()
                # Special handling for matrix rows
                if self._current_section == "matrix" and key.startswith("row"):
                    # Parse list of integers for matrix rows
                    try:
                        parsed_value = [int(x.strip()) for x in value.split()]
                    except ValueError:
                        raise ValueError(f"Invalid integer list for {key}")
                else:
                    # Parse single values (int, float, or string)
                    parsed_value = self._parse_single_value(value)

                self._config[self._current_section][key] = parsed_value

    @property
    def number_of_channels(self) -> int:
        return self._number_of_channels

    def _parse_single_value(self, value: str) -> Any:
        """
        Parse a single value to the most appropriate type.

        Parameters
        ----------
        value: str
             Value to parse

        Returns
        -------
        int | float | string
            Parsed value as int, float, or string
        """
        value = value.strip()

        # Try parsing as int
        try:
            return int(value)
        except ValueError:
            # Try parsing as float
            try:
                return float(value)
            except ValueError:
                # Return as string
                return value

    def get_config(self) -> dict[str, dict[str, Any]]:
        """
        Retrieve the parsed configuration.

        Returns
        -------
        dict[str, dict[str, str | int | float | list[int]]]
             Nested dictionary of configuration
        """
        return self._config

    def get_section(self, section: str = "root") -> dict[str, Any]:
        """
        Retrieve a specific section of the configuration.

        Parameters
        ----------
        section: str
             Section name (defaults to 'root')

        Returns
        -------
        dict[str, str | int | float | list[int]]
            Dictionary of key-value pairs in the section

        Raises
        ------
        KeyError
            If the section does not exist
        """
        return self._config[section]

    def get_channel_name_map(self) -> dict[str, str]:
        """
        Get a mapping of channel sensor name -> channel section name

        Returns
        -------
        dict[str, str]
        """
        channels = [k for k in self._config.keys() if k.startswith("channel")]
        channel_map = dict([(str(self.get_section(c)["name"]), c) for c in channels])
        return channel_map

    def get_channel_config(self, name: str) -> dict[str, Any] | None:
        """
        Get the configuration of a channel

        Parameters
        ----------
        name : str
            name of channel

        Returns
        -------
        dict[str, str | int | float | list[int]] | None
            a dictionary with configuration values or None if name is not found.
        """
        channel_map = self.get_channel_name_map()
        if name in channel_map.keys():
            return self.get_section(channel_map[name])
        else:
            return None
