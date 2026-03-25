import os
from pathlib import Path
import requests
import shutil
import sys
import tempfile
from urllib.parse import urlparse
from zipfile import ZipFile, BadZipFile

from turban import logger_manager

logger = logger_manager.get_logger(__name__)

DATA_DOWNLOAD_LINK = "https://share.hereon.de/index.php/s/D89zzgAbdLcCc7m/download"
ARCHIVE_NAME = "Turban"  # Top level directory name of the link


def copytree(src, dst, *, overwrite=True):
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_root = dst.joinpath(rel)
        target_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            src_file = Path(root) / f
            dst_file = target_root / f
            if dst_file.exists() and not overwrite:
                continue
            shutil.copy2(src_file, dst_file)  # preserves metadata


class FilePaths:
    """FilePaths

    A class for managing downloading data files for test purposes from a remote server.

    Intended use:

    >>> filepaths = FilePaths()
    >>> atomix_benchmark_baltic_fpath = filepaths.add("data/process/shear/MSS_Baltic.nc")
    >>> # and more paths you want to speficy
    >>> filepaths.download_data_if_necessary()

    """

    def __init__(self) -> None:
        self.filepaths: list[str] = []
        self.top_level: Path = Path(__file__).resolve().parent.parent.parent
        self.url: str = ""
        self._is_data_downloaded = False
        
    def add(self, path: str | Path) -> str:
        """Adds given string or Path object to the registry.

        Parameters
        ----------
        path : str or Path
            Name of a path to add to the registry

        Returns: str
            Path name as a string
        """
        file_path = self.top_level / path
        self.filepaths.append(file_path)
        return str(file_path)

    def auto_download_data_if_necessary(self) -> None:
        '''Download files if one or more data are missing, if and only if, the environment
           variable TURBAN_AUTO_DOWNLOAD_TEST_FILES is set to 1.
        '''
        flag = os.getenv('TURBAN_AUTO_DOWNLOAD_TEST_FILES')
        if not flag is None and flag.strip()=='1':
            logger.info(f"Environment variable TURBAN_AUTO_DOWNLOAD_TEST_FILES is set. Checking need for download.")
            self.download_data_if_necessary()
            
    def download_data_if_necessary(self):
        """Download if one or more data files are missing

        This method checks if all required files are existing on the current system,
        and in case one or more are failing, the data files are retrieved from a central
        public data file server.

        Data download, and extraction are done in a temporary directoy, before being moved
        to the expected location.
        """
        if self._is_data_downloaded:
            return
        download_required = False
        for p in self.filepaths:
            if not p.exists():
                logger.info(f"File {str(p)} is missing.")
                download_required = True
        if download_required:
            self.download_data()
        self._is_data_downloaded = download_required # if downloaded this session, we will not do so again.
            
    def download_data(self) -> None:
        """Downloads test and benchmark data files from external server"""
        logger.info("Downloading...")
        url = self.url or DATA_DOWNLOAD_LINK
        with tempfile.TemporaryDirectory(prefix="turban_", dir=None) as tmpdir:
            dest_dir = Path(tmpdir)
            zip_file = self.download_as_zip(url, dest_dir)
            logger.debug(f"Downloaded {zip_file}.")
            self.safe_extract_zip(zip_file, dest_dir)
            copytree(dest_dir / ARCHIVE_NAME, self.top_level)
        logger.info("Download completed.")
        
    def download_as_zip(self, url: str, dest_dir: str | Path, timeout: int = 30) -> str:
        """Download URL and write as a zip file

        Parameters
        ----------
        url : str
            url for remote site
        dest_dir : str | Path
            path where to download the zip file

        Returns
        -------
        str:
            name of zip file
        """
        headers = {"User-Agent": "curl/7.88.1"}
        os.makedirs(dest_dir, exist_ok=True)
        local_name = str(Path(dest_dir) / "data.zip")
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            tmp = local_name + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, local_name)
        return local_name

    def safe_extract_zip(self, zip_path, target_dir) -> None:
        """
        Extract zip_path into target_dir while preventing path traversal attacks. (AI generated method)

        Parameters
        ----------
        zip_path : str | Path
            path to zip file to be extracted
        target_dir : str | Path
            path to directory into which the zip file should be extracted.
        """
        zip_path = Path(zip_path)
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path, "r") as z:
            for member in z.namelist():
                member_path = target_dir / member
                # Prevent path traversal: ensure the resolved path is inside target_dir
                if not str(member_path.resolve()).startswith(
                    str(target_dir.resolve()) + os.sep
                ):
                    raise RuntimeError(f"Unsafe zip entry detected: {member}")
                z.extractall(path=target_dir)


# NOTE: Add all paths to data files used in the tests using the format below.
#       Once data are added, you need to make sure that those files are also available
#       from the share.hereon.de server (for the time being). If you have data to add,
#       please make sure they find there way into the public data repository. Contact
#       lucas dot merckelbach at hereon dot de for that.
#
filepaths = FilePaths()
atomix_benchmark_baltic_fpath = filepaths.add("data/process/shear/MSS_Baltic.nc")
atomix_benchmark_faroe_fpath = filepaths.add(
    "data/process/shear/VMP2000_FaroeBankChannel.nc"
)
atomix_benchmark_baltic_mrd_fpath = filepaths.add("data/instruments/mss/SH2_0330.MRD")
mss_mrd_fpath = filepaths.add("data/instruments/mss/Nien0020.MRD")
mss_probeconf_json_fpath = filepaths.add(
    "data/instruments/mss/probeconf_mss053_2024.json"
)
mss_utemp_mrd_fpath = filepaths.add("data/instruments/mss/probeconf_mss053_2024.json")

microrider_data_directory = filepaths.add("data/instruments/microrider")
# make sure that individual files are present:
filepaths.add("data/instruments/microrider/DAT_058.mat")
filepaths.add("data/instruments/microrider/DAT_058.P")
filepaths.add("data/instruments/microrider/data_0413.mat")
filepaths.add("data/instruments/microrider/data_0413.p")
filepaths.add("data/instruments/microrider/setupstring_0413.txt")
filepaths.add("data/instruments/microrider/setupstring_058.txt")

# imported as module: checks TURBAN_AUTO_DOWNLOAD_TEST_FILES environment variable to be set before downloading
filepaths.auto_download_data_if_necessary()
    
