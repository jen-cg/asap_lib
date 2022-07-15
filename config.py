import os

"""
Paths to data and default parameters 
(may be user specific)   

"""

# -----------------------------------------------------------------------------------------------------------------------
# CHANGE ME: Path to the overarching directory where your spectra are stored (BE SURE THIS DIRECTORY ALREADY EXISTS)
dataDir = '/arc5/home/jglover/Data/'

# -----------------------------------------------------------------------------------------------------------------------
# Paths to model atmospheres
spherical_model_atm_path = os.path.join(os.getcwd(), 'asap_lib/data/grids/standard/sphere/')

plane_parallel_model_atm_path = os.path.join(os.getcwd(), 'asap_lib/data/grids/standard/plane/')

# -----------------------------------------------------------------------------------------------------------------------
# Default Parameters
default_vmicro = 1.8
