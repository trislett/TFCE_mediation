import Cython.Compiler.Main

import numpy
from numpy.distutils.command import build_src
from numpy.distutils.misc_util import Configuration

from distutils.dir_util import mkpath

import os

PACKAGE_NAME = "tfce_mediation"

def configuration(parent_package = "", top_path = None):
	from numpy.distutils.misc_util import Configuration
	CONFIG = Configuration(PACKAGE_NAME, 
		parent_name = parent_package, 
		top_path = top_path)

  	CONFIG.add_data_dir("adjacency_sets")

	CONFIG.add_subpackage("tmanalysis")
	CONFIG.add_subpackage("tools")

	CONFIG.add_extension("tfce", 
		sources = ["tfce.pyx"],
		include_dirs = ["lib", numpy.get_include()],
		language = "c++",
		extra_compile_args = ["-std=c++11", "-Wno-unused", "-g"])

	CONFIG.add_extension("cynumstats", 
		sources = ["cynumstats.pyx"],
		include_dirs = [numpy.get_include()],
		language = "c++",
		extra_compile_args = ["-std=c++11", "-Wno-unused", "-g"])

	CONFIG.add_extension("adjacency", 
		sources = ["adjacency.pyx"],
		include_dirs = ["lib", numpy.get_include()],
		language = "c++",
		extra_compile_args = ["-std=c++11", "-Wno-unused", "-g"])

	def cythonize(self, base, ext_name, source, extension):
		target_ext = '.cpp'

		target_dir = CONFIG.get_build_temp_dir()
		target_dir = os.path.join(target_dir, "pyrex")
		for package_name in extension.name.split('.')[:-1]:
			target_dir = os.path.join(target_dir, package_name)
		
		new_sources = []
		cython_targets = {}

		for source in extension.sources:
			(base, ext) = os.path.splitext(os.path.basename(source))
			new_sources.append(os.path.join(target_dir, base + target_ext))
			cython_targets[source] = new_sources[-1]

		module_name = extension.name
			
		for source in extension.sources:
			target = cython_targets[source]
			mkpath(os.path.dirname(target))
			options = Cython.Compiler.Main.CompilationOptions(
				defaults = Cython.Compiler.Main.default_options,
				include_path = extension.include_dirs,
				output_file = target,
				verbose = True,
				cplus = True)
			result = Cython.Compiler.Main.compile([source], 
				options = options, full_module_name = module_name)

		if len(new_sources) == 1:
			return new_sources[0]
		return new_sources
		
	build_src.build_src.generate_a_pyrex_source = cythonize

	CONFIG.make_config_py()
	return CONFIG

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
