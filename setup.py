from setuptools import setup, find_packages
import re

DESCRIPTION = 'Redshift finding algorithm'
LONG_DESCRIPTION = 'Find the redshift of high redshift radio galaxies'

with open('./zfinder/__init__.py', 'r') as f:
    contents = f.read()
    VERSION = re.search(r"__version__ = '(.+)'", contents).group(1)

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="zfinder",
    version=VERSION,
    author="Daniel Lyon",
    author_email="daniellyon31@gmail.com",
    description=DESCRIPTION,
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