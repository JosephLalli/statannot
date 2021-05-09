from distutils.core import setup
from setuptools import find_packages
import re

with open("readme.md", "r") as f:
    long_description = f.read()

VERSIONFILE = "statannot/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", verstrline, re.M)
if match:
    version = match.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name="statannot",
    version=version,
    author="Marc Weber",
    author_email="webermarcolivier@gmail.com",
    description="add statistical annotations on an existing boxplot/barplot generated by seaborn.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/webermarcolivier/statannot",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # install_requires=open("requirements.txt").readlines(),
    python_requires='>=3.5',
)
