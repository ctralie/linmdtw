import sys
import os
import platform

from setuptools import setup
from setuptools.extension import Extension

# Ensure Cython is installed before we even attempt to install linmdtw
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org or install it with `pip install Cython`")
    sys.exit(1)

## Get version information from _version.py
import re
VERSIONFILE="linmdtw/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Use README.md as the package long description  
with open('README.md') as f:
    long_description = f.read()

class CustomBuildExtCommand(build_ext):
    """ This extension command lets us not require numpy be installed before running pip installing
        build_ext command for use when numpy headers are needed.
    """

    def run(self):
        # Import numpy here, only when headers are needed
        import numpy
        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())
        # Call original build_ext command
        build_ext.run(self)

extra_compile_args = []
extra_link_args = []

if platform.system() == "Darwin":
    extra_compile_args.extend([
        "-mmacosx-version-min=10.9"
    ])
    extra_link_args.extend([
        "-mmacosx-version-min=10.9"
    ])

ext_modules = Extension(
    "dynseqalign",
    sources=["linmdtw/dynseqalign.pyx"],
    define_macros=[
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++"
)


setup(
    name="linmdtw",
    version=verstr,
    description="A linear memory, exact, parallelizable algorithm for dynamic time warping in arbitrary Euclidean spaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chris Tralie",
    author_email="ctralie@alumni.princeton.edu",
    license='Apache2',
    packages=['linmdtw'],
    ext_modules=cythonize(ext_modules, include_path=['linmdtw']),
    install_requires=[
        'Cython',
        'numpy',
        'matplotlib',
        'scipy',
        'numba'
    ],
    extras_require={
        'testing': [ # `pip install -e ".[testing]"``
            'pytest'  
        ],
        'docs': [ # `pip install -e ".[docs]"`
            'linmdtw_docs_config'
        ],
        'examples': []
    },
    cmdclass={'build_ext': CustomBuildExtCommand},
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    keywords='dynamic time warping, alignment, fast dtw, synchronization, time series, music information retrieval, audio analysis'
)
