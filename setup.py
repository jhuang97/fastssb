"""A setuptools based setup module.
See https://packaging.python.org/en/latest/distributing.html
Addapted from https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from codecs import open
from os import path

from setuptools import setup , find_packages


# To use a consistent encoding
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
#with open(path.join(here, 'ncempy/long_description.rst'), encoding='utf-8') as f:
#    long_description = f.read()

setup(
    name='fastssb',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',

    description='realtime-ptychography Python Package',
    long_description='long_description',

    # The project's main homepage.
    url='https://github.com/PhilippPelz/realtime_ptychography',

    # Author details
    author='P. Pelz',
    author_email='philipp.pelz@berkeley.edu',

    # Choose your license
    license='GPLv3+, MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    # What does your project relate to?
    keywords='electron microscopy',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'matplotlib', 'h5py', 'ncempy', 'numba', 'scikit-image', 'matplotlib-scalebar', 'tifffile'],

    # Before installing fastssb, CUDA must be installed.

    # cupy may be installed manually, using the wheel that matches your cuda version, e.g.,
    # $ pip install cupy-cuda117
    # for CUDA v11.7

    # alternatively,
    # $ pip install cupy
    # will install from source. This may take a long time.

    # also recommend installing pytorch manually in order to install the one for the right CUDA version, for example
    # $ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    #extras_require={
    #},

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={}
    include_package_data=False,

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': ['fastssb = fastssb.cli:main'],
    },
)