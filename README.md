[![Build Status](https://api.travis-ci.org/ctralie/linmdtw.svg?branch=master)](https://api.travis-ci.org/ctralie/linmdtw)
[![codecov](https://codecov.io/gh/ctralie/linmdtw/branch/master/graph/badge.svg)](https://codecov.io/gh/ctralie/linmdtw/)

# Parallelizable Dynamic Time Warping with Linear Memory

## Dependencies
* cython
* numba
* librosa
* pycuda

To get started, type
~~~~~ bash
python setup.py build_ext --inplace
~~~~~

Run the file Orchestral.py to download and run all alignment experiments