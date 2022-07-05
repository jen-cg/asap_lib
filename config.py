import os

"""
Paths to data and default parameters 
(may be user specific)   

"""

# -----------------------------------------------------------------------------------------------------------------------
# Paths to model atmospheres
spherical_model_atm_path = os.path.join( os.getcwd(), 'Data/grids/standard/sphere/')
#'/arc5/home/jglover/GRACES/Code/MARCS/grids/standard/sphere/'  # CHANGED

plane_parallel_model_atm_path = os.path.join( os.getcwd(), 'Data/grids/standard/plane/')
#'/arc5/home/jglover/GRACES/Code/MARCS/grids/standard/plane/'  # CHANGED

# -----------------------------------------------------------------------------------------------------------------------
# Default Parameters
default_vmicro = 1.8
