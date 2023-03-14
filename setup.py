import io
import os
import re
from os import path
from setuptools import setup, find_packages
from pkg_resources import parse_requirements
from codecs import open     # To use consistent encodings

PACKAGE_NAME = 'face_alignment'
PACKAGE_PATH = PACKAGE_NAME
REQUIREMENTS_FILE_PATH = 'requirements.txt'


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = find_version(PACKAGE_NAME, '__init__.py')

with open(REQUIREMENTS_FILE_PATH, 'r') as requirements_file:
    text = requirements_file.read()
requirements_text = re.sub(r'^\s*\-e\s+([^=]+)=([^\n]+)', r'\2 @ \1=\2', text, flags=re.MULTILINE)


setup(
    name=PACKAGE_NAME,
    version='0.0.8',
    packages=find_packages(
        exclude=(
            'test',
        ),
    ),
)
