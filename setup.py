from setuptools import setup, find_packages
import re
import os
import codecs

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'zfinder/__init__.py'), 'r') as f:
    VERSION = re.search(r"__version__ = '(.+)'", f.read()).group(1)

with open(os.path.join(here, 'requirements.txt'), 'r') as f:
    requirements = f.read().splitlines()
    
with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    LONG_DESCRIPTION = "\n" + fh.read()

setup(
    name="zfinder",
    version=VERSION,
    author="Daniel Lyon",
    author_email="daniellyon31@gmail.com",
    description='Redshift finding algorithm',
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=requirements,
    keywords=['python', 'visualisation', 'redshift', 'galaxy-evolution', 'cosmology', 
              'epoch-of-reionisation', 'radio-galaxies', 'active-galactic-nuclei',
              'high-redshift', 'galaxies', 'black-holes'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)