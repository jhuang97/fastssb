# GPU-accelerated single side-band ptychography

Python code to calculate a phase image from the bright field disk of atomic resolution 4D-STEM data using the single side-band method.

## Acknowledgements
Philipp Pelz authored the original GPU-accelerated single side-band code with post-processing aberration correction ([Github repo](https://github.com/PhilippPelz/realtime_ptychography), [paper](https://ieeexplore.ieee.org/abstract/document/9664587)). [Peter Ercius](https://foundry.lbl.gov/about/staff/peter-ercius/) helped me to understand and use this code.

I built upon this foundation by reorganizing the code, rewriting the data pre-processing part, and changing the aberration correction user interface somewhat.
## Features
* Fast: The code takes less than a minute to import and preprocess a 4D-STEM data set, and less than 10 seconds (on my computer at least) to compute a single side-band reconstruction.
* I've written pre-processing code to support data from both the EMPAD and the 4D Camera from NCEM (see the provided Jupyter notebooks).
* Post-processing aberration correction.  You can manually correct for defocus in your 4D-STEM data, allowing for you to quickly check your defocused 4D-STEM data sets. (Can also manually correct for geometric aberrations up to 3rd order.)
* Plots slices of $G(K_f, Q_p)$, which may let you better understand the probe aberrations in your 4D data set.
## Limitations
* You have to manually put in experimental parameters like convergence angle, probe step size, etc.
* Limitations of single side-band ptychography:  Assumes a weak phase object. Spatial resolution is limited by convergence angle; it can be a bit better than ADF-STEM but is not as good as iterative, full-field ptychography.
* Though the crop_bin_sparse_to_dense function uses less GPU memory, it still requires a lot of RAM (10s of GB for some 4D Camera scans) -- there is room for improvement here.
## Requirements
* Needs a Nvidia GPU and CUDA installed.
## Installation
1. Install [cupy](https://docs.cupy.dev/en/stable/install.html).  The command you need to run likely depends on the version of CUDA you have, e.g.,
    pip install cupy-cuda117
for CUDA v11.7.
2. Install [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/user_install.html).
3. Install this package using pip.  Open a terminal in this folder, and run
    pip install .
or
    pip install -e .
This should automatically install this code as a Python package, along with all dependencies not already listed.