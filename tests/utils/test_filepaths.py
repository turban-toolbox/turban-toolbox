import os
import pytest

import turban
from turban.utils import filepaths

logger = turban.logger_manager.get_logger(__name__)


@pytest.fixture
def monkeypatch_test(monkeypatch):
    monkeypatch.setattr(filepaths, "ARCHIVE_NAME", "TurbanPytest")
    monkeypatch.setattr(filepaths, "TURBAN_AUTO_DOWNLOAD_TEST_FILES", "TURBAN_TEST")
    monkeypatch.setattr(filepaths.filepaths, "url", "https://share.hereon.de/index.php/s/C8T5fNfX2PrZtDL/download")

@pytest.fixture
def setupandteardown():
    logger.debug("setupandteardown")
    directory = "test"
    path = os.path.join(directory, "test.txt")
    os.remove(path) if os.path.exists(path) else None
    try:
        os.environ.pop(filepaths.TURBAN_AUTO_DOWNLOAD_TEST_FILES)
    except KeyError:
        pass
    fp = filepaths.filepaths
    fp.filepaths.clear()
    fp.add(path)
    fp._is_data_downloaded=False # <= needs to be set as we assume a new session each test.
    yield path, fp
    try:
        os.remove(path)
        os.rmdir(directory)
    except:
        pass
    
def test_download_data_if_necessary(monkeypatch_test, setupandteardown):
    path, fp = setupandteardown
    fp.download_data_if_necessary()
    result = os.path.exists(path)
    assert result

def test_auto_download_data_if_necessary(monkeypatch_test, setupandteardown):
    path, fp = setupandteardown
    fp.auto_download_data_if_necessary()
    result = not os.path.exists(path) # the environment variable should not be set, so nothing will be downloaded.
    assert result

def test_auto_download_data(monkeypatch_test, setupandteardown):
    path, fp = setupandteardown
    os.environ[filepaths.TURBAN_AUTO_DOWNLOAD_TEST_FILES] = '1'
    fp.auto_download_data_if_necessary()
    result = os.path.exists(path) # the environment variable should be set, so we download stuff
    assert result
    
