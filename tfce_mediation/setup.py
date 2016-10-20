import Cython.Compiler.Main

import numpy
from numpy.distutils.command import build_src

PACKAGE_NAME = "tfce_mediation"

def cythonize(self, base, ext_name, source, extension):
  compilationOptions = Cython.Compiler.Main.CompilationOptions(
    defaults = Cython.Compiler.Main.default_options,
    include_path = extension.include_dirs,
    cplus = True)
  compilationResult = Cython.Compiler.Main.compile([source], compilationOptions)
  return compilationResult.values()[0].c_file
build_src.build_src.generate_a_pyrex_source = cythonize

def configuration(parent_package = "", top_path = None):
  from numpy.distutils.misc_util import Configuration
  CONFIG = Configuration(PACKAGE_NAME, 
    parent_name = parent_package, 
    top_path = top_path)

  CONFIG.add_subpackage("vertex")
  CONFIG.add_subpackage("voxel")

  CONFIG.add_extension("tfce", 
    sources = ["tfce.pyx"],
    include_dirs = ["lib", numpy.get_include()],
    language = "c++",
    extra_compile_args = ["-std=c++11", "-Wno-unused", "-g"])

  CONFIG.add_extension("stats", 
    sources = ["stats.pyx"],
    include_dirs = [numpy.get_include()],
    language = "c++",
    extra_compile_args = ["-std=c++11", "-Wno-unused", "-g"])

  CONFIG.make_config_py()
  return CONFIG

if __name__ == '__main__':
  from numpy.distutils.core import setup
  setup(**configuration(top_path='').todict())
