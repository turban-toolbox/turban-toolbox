import argparse
from turban.utils.filepaths import filepaths
from turban.utils.logging import LoggerManager


def download_datafiles():
    logger_manager = LoggerManager()
    logger_manager.set_level("info")
    logger = logger_manager.get_logger(__name__)
    parser = argparse.ArgumentParser(
        description="Download test and benchmark data files for the turban-toolbox"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download of files, even if all files appear to be present.",
    )
    args = parser.parse_args()
    if not filepaths._is_data_downloaded:
        if args.force:
            logger.info("Forced to download data files ...")
            filepaths.download_data()
        else:
            logger.info("Checking whether to download data files ...")
            filepaths.download_data_if_necessary()
            if not filepaths._is_data_downloaded:
                logger.info("No action required this time.")
