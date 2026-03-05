import os
import pytest

from  tests import filepaths

@pytest.fixture
def download_test(monkeypatch):
    monkeypatch.setattr(filepaths, "ARCHIVE_NAME", "test")


def test_download_data_if_necessary(download_test):
    try:
        os.unlink("test.txt")
    except OSError:
        pass
    fp = filepaths.filepaths
    fp.filepaths.clear()
    fp.add("test.txt")
    fp.url="https://share.hereon.de/index.php/s/Y2tYW2w28zKLpk3/download"
    fp.download_data_if_necessary()
    result = os.path.exists("test.txt")
    try:
        os.unlink("test.txt")
    except OSError:
        pass
    assert result
