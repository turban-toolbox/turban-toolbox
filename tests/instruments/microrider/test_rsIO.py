from itertools import chain
import io
import os
import pytest
import requests
import shutil
import zipfile

import numpy as np
from scipy.io import loadmat

from turban.instruments.microrider import channels, rsIO, rsConfig_parser
from turban.utils.filepaths import microrider_data_directory

datadir = microrider_data_directory
data_url = "https://share.hereon.de/index.php/s/AqMdY8Q47FPQHQR/download"
archive_name = "TurbanData"

# data_url = "https://share.hereon.de/index.php/s/Ek2JGYnZWgPx8FR/download"
# archive_name = "test"


def download_and_extract_zip(url, output_filename, extract_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(output_filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        with zipfile.ZipFile(output_filename, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Successfully downloaded and extracted {url}")
        archive_path = os.path.join(extract_path, archive_name)
        fns = os.listdir(archive_path)
        for fn in fns:
            src = os.path.join(archive_path, fn)
            dst = os.path.join(datadir, fn)
            shutil.move(src, dst)
        os.unlink(output_filename)
        os.rmdir(archive_path)
    except requests.RequestException as e:
        print(f"Download error: {e}")
    except zipfile.BadZipFile:
        print("Error: Invalid zip file")

def load_odas_reference_data(mat_path):
    # ODAS reference output
    mdata = loadmat(
        mat_path,
        squeeze_me=True,
        mat_dtype=False,
        chars_as_strings=True,
        simplify_cells=True,
    )
    return mdata

def test_presence_of_test_data():
    fns = os.listdir(datadir)
    expected_datafiles = "DAT_058.P data_0413.p DAT_058.mat data_0413.mat".split()
    present = True
    for fn in expected_datafiles:
        if fn not in fns:
            present = False
            break
    if not present:
        download_and_extract_zip(
            data_url, output_filename="/tmp/TurbanData.zip", extract_path="/tmp"
        )
        present = True
        fns = os.listdir(datadir)
        for fn in expected_datafiles:
            if fn not in fns:
                present = False
                break
        assert present
    assert present


def test_header_data_058():
    header_parser = rsIO.HeaderParser()
    with open(os.path.join(datadir, "DAT_058.P"), "rb") as fp:
        data = header_parser.parse(fp)
    assert (
        data.file_number == 58
        and data.year == 2018
        and data.data_record_size == 4672
        and data.data_size == 4608
    )


def test_header_data_0413():
    header_parser = rsIO.HeaderParser()
    with open(os.path.join(datadir, "data_0413.p"), "rb") as fp:
        data = header_parser.parse(fp)
    assert (
        data.file_number == 413
        and data.year == 2025
        and data.data_record_size == 4672
        and data.data_size == 4608
    )


def test_check_for_bad_blocks_058():
    header_parser = rsIO.HeaderParser()
    with open(os.path.join(datadir, "DAT_058.P"), "rb") as fp:
        number_of_records = header_parser.check_for_bad_blocks(fp)
    assert number_of_records == 11593


def test_check_for_bad_blocks_0413():
    header_parser = rsIO.HeaderParser()
    with open(os.path.join(datadir, "data_0413.p"), "rb") as fp:
        number_of_records = header_parser.check_for_bad_blocks(fp)
    assert number_of_records == 483


@pytest.fixture
def corrupted_0413(tmp_path):
    """Returns a function that writes a modified copy of data_0413.p and returns its path"""
    fn = os.path.join(datadir, "data_0413.p")
    with open(fn, "rb") as fp:
        header_data = rsIO.HeaderParser().parse(fp)
        fp.seek(0)
        content = fp.read()
    # Record 0 is the setup string, followed by the data records.
    first_data_record = rsIO.HeaderEnum.HeaderSize + header_data.setupfile_size
    record_size = rsIO.HeaderEnum.HeaderSize + header_data.data_size * rsIO.HeaderEnum.WordSize

    def write(modify):
        path = tmp_path / "corrupted.p"
        path.write_bytes(modify(content, first_data_record, record_size))
        return path

    return write


@pytest.mark.parametrize(
    "modify",
    [
        # cut off in the middle of the data block of the last record
        lambda c, start, size: c[:-100],
        # cut off in the middle of the header of the last record
        lambda c, start, size: c[: -size + 50],
    ],
    ids=["in_data", "in_header"],
)
def test_check_for_bad_blocks_truncated(corrupted_0413, modify):
    path = corrupted_0413(modify)
    with open(path, "rb") as fp:
        with pytest.raises(rsIO.uriderException, match="File seems truncated."):
            rsIO.HeaderParser().check_for_bad_blocks(fp)


def test_check_for_bad_blocks_missing_record(corrupted_0413):
    # remove the 5th data record
    path = corrupted_0413(
        lambda c, start, size: c[: start + 4 * size] + c[start + 5 * size :]
    )
    with open(path, "rb") as fp:
        with pytest.raises(rsIO.uriderException, match="Missing record"):
            rsIO.HeaderParser().check_for_bad_blocks(fp)


#
# Test the ChannelMatrix class
#


def test_channel_config():
    channel_config = channels.ChannelConfig(name="ch255", id=255, type="")
    assert channel_config.name == "ch255"
    assert channel_config.id == 255
    assert channel_config.type == ""
    channel_config.update("name", "CH255")
    assert channel_config.name == "CH255"
    with pytest.raises(
        AttributeError, match="ChannelConfig has no attribute not_exsting."
    ):
        channel_config.update("not_exsting", 13)


def test_channel_config_is_set():
    channel_config = channels.ChannelConfigThermistor(name="T1", id=12, type="therm")
    assert channel_config.is_set("name")
    assert not channel_config.is_set("a")
    with pytest.raises(
        AttributeError, match="ChannelConfigThermistor has no attribute a0"
    ):
        channel_config.is_set("a0")
    channel_config.update("a", 1)
    assert channel_config.is_set("a")


#
# Test MicroRiderData class
#
@pytest.fixture
def microrider_data(request):
    variable = request.param
    match variable:
        case "058":
            fn = os.path.join(datadir, "DAT_058.P")
            mfn = os.path.join(datadir, "DAT_058.mat")
            fns = os.path.join(datadir, "setupstring_058.txt")
        case "0413":
            fn = os.path.join(datadir, "data_0413.p")
            mfn = os.path.join(datadir, "data_0413.mat")
            fns = os.path.join(datadir, "setupstring_0413.txt")
    mdata = load_odas_reference_data(mfn)
    with open(fns, "r") as fp:
        setupstring = "\n".join(fp.readlines())
    microrider_config = rsConfig_parser.MicroRiderConfig()
    microrider_config.parse(setupstring)
    full_path = os.path.realpath(fn)
    header_parser = rsIO.HeaderParser()
    with open(fn, "rb") as fd:
        header_data = header_parser.parse(fd)
        n_records = header_parser.check_for_bad_blocks(fd)
        header_info = header_parser.get_header_info(header_data, n_records)
    return dict(
        microrider_config=microrider_config,
        full_path=full_path,
        header_info = header_info,
        header_data = header_data,
        mdata = mdata
    )


def parse_datetime_string(s, separator):
    try:
        fields = s.split(separator)
        value = f"{int(fields[0])} {int(fields[1])} {float(fields[2]):.3f}"
    except:
        value = ""
    return value

@pytest.mark.parametrize("microrider_data", ["058", "0413"], indirect=True)
def test_microriderdata_header(microrider_data):
    microrider_config = microrider_data["microrider_config"]
    header_info = microrider_data["header_info"]
    mdata = microrider_data["mdata"]
    assert (
        np.abs(header_info.t_fast.max() - mdata["t_fast"].max()) < 0.01
    )  # matlab/octave only stores 2 decimal positions
    assert len(header_info.t_fast) == len(mdata["t_fast"])
    assert np.abs(header_info.t_slow.max() - mdata["t_slow"].max() ) < 0.01
    assert len(header_info.t_slow) == len(mdata["t_slow"])
    assert (parse_datetime_string(header_info.time, ":") == parse_datetime_string(mdata["time"], ":")
            and 
            parse_datetime_string(mdata["time"], ":")
            )
    assert (parse_datetime_string(header_info.date, "-") == parse_datetime_string(mdata["date"], "-")
            and 
            parse_datetime_string(mdata["date"], "-")
            )


@pytest.fixture
def microrider_channel_data(request):
    variable = request.param
    match variable:
        case "058":
            fn = os.path.join(datadir, "DAT_058.P")
            mfn = os.path.join(datadir, "DAT_058.mat")
        case "0413":
            fn = os.path.join(datadir, "data_0413.p")
            mfn = os.path.join(datadir, "data_0413.mat")
    data = rsIO.read_p_file(fn)
    mdata = load_odas_reference_data(mfn)
    return (data, mdata)

# ODAS stores the deconvolved channels on the fast time base under these names
odas_names = dict(T1="T1_fast", T2="T2_fast", P="P_fast")

@pytest.mark.parametrize("microrider_channel_data", ["058", "0413"], indirect=True)
def test_channel_data(microrider_channel_data):
    data, mdata = microrider_channel_data
    channels_to_test = ['T1', 'T2', 'P', 'Gnd', 'Ax', 'Ay', 'sh1', 'sh2', 'PV', 'V_Bat', 'Incl_Y', 'Incl_X', 'Incl_T', 'EMC_Cur', 'U_EM']
    # The DAT_058 reference was made with U_EM coefficients a=-24.4605, b=151.341,
    # where the setup string has a=-24.46052, b=151.3413 (diff up to 1.6e-6 m/s).
    for name in channels_to_test:
        v = data[name]
        v_ref = mdata[odas_names.get(name, name)]
        if name == "U_EM":
            atol=2e-6 # see comment above.
        else:
            atol=2e-12 # close to machine precision
        assert v.shape == v_ref.shape, name
        assert np.allclose(v, v_ref, rtol=0, atol=atol), name


def test_user_channel_config_058():
    # Supply the U_EM coefficients with which the DAT_058 reference was made, so
    # that the user-supplied config overrides the one in the setup string.
    fn = os.path.join(datadir, "DAT_058.P")
    mfn = os.path.join(datadir, "DAT_058.mat")
    u_em_config = channels.ChannelConfigU_EM(name="U_EM", a=-24.4605, b=151.341)
    data = rsIO.read_p_file(fn, channel_configs=[u_em_config])
    mdata = load_odas_reference_data(mfn)
    assert data["U_EM"].shape == mdata["U_EM"].shape
    # rtol=0: a relative tolerance would hide the 1.6e-6 m/s difference without the override.
    assert np.allclose(data["U_EM"], mdata["U_EM"], rtol=0, atol=2e-12)
    

    
