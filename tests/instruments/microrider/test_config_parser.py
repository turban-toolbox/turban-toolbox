import hashlib
import os
import pytest
import json
from turban.instruments.microrider import rsConfig_parser
from tests.filepaths import microrider_data_directory

datadir = microrider_data_directory

@pytest.fixture
def setupstring_data(request):
    variable = request.param
    match variable:
        case "058":
            fn = os.path.join(datadir, "setupstring_058.txt")
            hash_value = "f4f5c67b547c89f25ae0778c5194428f"
            number_of_channels = 19
        case "0413":
            fn = os.path.join(datadir, "setupstring_0413.txt")
            hash_value = "81a36f63cf537a619d9124f7d3d86cf9"
            number_of_channels = 18
    with open(fn, "r") as fp:
        setupstring = "\n".join(fp.readlines())
    return dict(
        setupstring=setupstring,
        hash_value=hash_value,
        number_of_channels=number_of_channels,
    )


@pytest.mark.parametrize("setupstring_data", ["058", "0413"], indirect=True)
def test_rsConfig_parser(setupstring_data):
    cp = rsConfig_parser.MicroRiderConfig()
    cp.parse(setupstring_data["setupstring"])
    s = json.dumps(cp.get_config()).encode()
    hash_value = hashlib.md5(s).hexdigest()
    assert hash_value == setupstring_data["hash_value"]


@pytest.mark.parametrize("setupstring_data", ["058", "0413"], indirect=True)
def test_get_channel_config(setupstring_data):
    cp = rsConfig_parser.MicroRiderConfig()
    cp.parse(setupstring_data["setupstring"])
    cfg = cp.get_channel_config("sh1")
    assert cfg["name"] == "sh1" and len(cfg.keys()) >= 8


@pytest.mark.parametrize("setupstring_data", ["058", "0413"], indirect=True)
def test_number_of_channels(setupstring_data):
    cp = rsConfig_parser.MicroRiderConfig()
    cp.parse(setupstring_data["setupstring"])
    assert cp.number_of_channels == setupstring_data["number_of_channels"]
