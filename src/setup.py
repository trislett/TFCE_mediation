import sys
import os
import shutil
import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# clean previous build
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if (name.endswith(".so")):
            os.remove(os.path.join(root, name))

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize([
        Extension("TFCE", 
                  sources=["TFCE.pyx"],
                  libraries=[],   
                  language="c++",             
                  include_dirs=[numpy.get_include()],
                  library_dirs=[],
                  extra_compile_args=["-fopenmp", "-O3", "-std=c++11"]
             )
        ])
)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize([
        Extension("Adjacency", 
                  sources=["Adjacency.pyx"],
                  libraries=[],   
                  language="c++",             
                  include_dirs=["geodesic", numpy.get_include()],
                  library_dirs=[],
                  extra_compile_args=["-fopenmp", "-O3", "-std=c++11"]
             )
        ])
)

setup(
    name = "cy_numstats",
    ext_modules = cythonize('cy_numstats.pyx'),  # accepts a glob pattern
)
