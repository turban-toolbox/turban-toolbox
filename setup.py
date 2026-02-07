from setuptools import setup, find_packages
import os

ROOT_DIR = "turban"
with open(os.path.join(ROOT_DIR, "VERSION")) as version_file:
    version = version_file.read().strip()

setup(version=version)
