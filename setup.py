import sys
import os
import platform

from setuptools import setup
from setuptools.extension import Extension

# Ensure Cython is installed before we even attempt to install Ripser.py
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org or install it with `pip install Cython`")
    sys.exit(1)

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
    sources=["dynseqalign.pyx"],
    define_macros=[
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++"
)


setup(
    name="dynseqalign",
    version="0.1",
    description="Alignment algorithms in C for numpy arrays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anonymous",
    author_email="anonymous@gmail.com",
    license='Apache2',
    packages=['ripser'],
    ext_modules=cythonize(ext_modules),
    install_requires=[
        'Cython',
        'numpy'
    ],
    extras_require={
        'testing': [ # `pip install -e ".[testing]"``
            'pytest'  
        ],
        'docs': [ # `pip install -e ".[docs]"`
            'sktda_docs_config'
        ],
        'examples': []
    },
    cmdclass={'build_ext': CustomBuildExtCommand},
)
