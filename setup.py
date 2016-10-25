import Cython.Compiler.Main

import os

import numpy
from numpy.distutils.core import setup
from numpy.distutils.command import build_src
from numpy.distutils.misc_util import Configuration

# import pdb

PACKAGE_NAME = "tfce_mediation"

def configuration(parent_package = "", top_path = None):
  from numpy.distutils.misc_util import Configuration
  CONFIG = Configuration(None)
  CONFIG.set_options(ignore_setup_xxx_py = True,
    assume_default_configuration = True,
    delegate_options_to_subpackages = True,
    quiet = True)
  CONFIG.add_data_dir(('tfce_mediation/adjacency_sets'))

  CONFIG.add_scripts(os.path.join("bin", PACKAGE_NAME))
  CONFIG.add_scripts(os.path.join("bin", 'tm_tools'))
  CONFIG.add_scripts(os.path.join("bin", 'submit_condor_jobs_file'))

  CONFIG.add_subpackage(PACKAGE_NAME)

  return CONFIG

setup(name = PACKAGE_NAME,
  maintainer = "Tristram Lett",
  maintainer_email = "tristram.lett@charite.de",
  description = "TFCE_mediation",
  long_description = "Fast regression and mediation analysis of vertex or voxel MR data with TFCE",
  url = "",
  download_url = "",
  license = "GNU General Public License v3 or later (GPLv3+)",
  configuration = configuration
)
