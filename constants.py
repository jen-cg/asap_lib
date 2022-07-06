from scipy import constants

# -----------------------------------------------------------------------------------------------------------------------
# Define the global variable s_o_l which is the speed of light in km/s
s_o_l = constants.c / 1000.

# -----------------------------------------------------------------------------------------------------------------------
# Solar abundances and errs from Asplund 2009 - may need to add more based on the lines you are interested in
"""
TODO: The listed abundances for carbon, nitrogen, oxygen do not match the values listed in asplund 
Note: Uranium abundance is from meteorites  
"""
AX_sol = {
    'Li I':  [1.05, 0.10],
    'C I':   [8.50, 0.06],
    'N I':   [7.86, 0.12],
    'O I':   [8.76, 0.07],
    'Na I':  [6.24, 0.04],
    'Mg I':  [7.60, 0.04],
    'Al I':  [6.45, 0.03],
    'Si I':  [7.51, 0.03],
    'S I':   [7.12, 0.03],
    'K I':   [5.03, 0.09],
    'Ca I':  [6.34, 0.04], 'Ca II': [6.34, 0.04],
    'Sc I':  [3.15, 0.04], 'Sc II': [3.15, 0.04],
    'Ti I':  [4.95, 0.05], 'Ti II': [4.95, 0.05],
    'V I':   [3.93, 0.08], 'V II':  [3.93, 0.08],
    'Cr I':  [5.64, 0.04], 'Cr II': [5.64, 0.04],
    'Mn I':  [5.43, 0.04], 'Mn II': [5.43, 0.04],
    'Fe I':  [7.50, 0.04], 'Fe II': [7.50, 0.04],
    'Co I':  [4.99, 0.07], 'Co II': [4.99, 0.07],
    'Ni I':  [6.22, 0.04], 'Ni II': [6.22, 0.04],
    'Cu I':  [4.19, 0.04], 'Cu II': [4.19, 0.04],
    'Zn I':  [4.56, 0.05],
    'Sr II': [2.87, 0.07],
    'Y II':  [2.21, 0.07],
    'Zr I':  [2.58, 0.04], 'Zr II': [2.58, 0.04],
    'Nb II': [1.46, 0.04],
    'Ru I':  [1.75, 0.08],
    'Rh I':  [0.91, 0.10],
    'Pd I':  [1.57, 0.10],
    'Ag I':  [0.94, 0.10],
    'Ba II': [2.18, 0.09],
    'La II': [1.10, 0.04],
    'Ce II': [1.58, 0.04],
    'Pr II': [0.72, 0.04],
    'Nd II': [1.42, 0.04],
    'Sm II': [0.96, 0.04],
    'Eu II': [0.52, 0.04],
    'Gd II': [1.07, 0.04],
    'Tb II': [0.30, 0.10],
    'Dy II': [1.10, 0.04],
    'Er II': [0.92, 0.05],
    'Tm II': [0.10, 0.04],
    'Hf II': [0.85, 0.04],
    'Os I':  [1.40, 0.08],
    'Ir I':  [1.38, 0.07],
    'Pb I':  [1.75, 0.10],
    'Th II': [0.02, 0.10],
    'U II':  [-0.54, 0.03]}

# -----------------------------------------------------------------------------------------------------------------------
# Dictionary to match atomic number with species - may need to add more based on the lines you are interested in
elem_dict = {
    3.0:  'Li I',
    6.0:  'C I',
    7.0:  'N I', 7.1: 'N II',
    8.0:  'O I',
    11.0: 'Na I',
    12.0: 'Mg I',
    13.0: 'Al I',
    14.0: 'Si I',
    15.0: 'P I',
    16.0: 'S I',
    19.0: 'K I',
    20.0: 'Ca I', 20.1: 'Ca II',
    21.0: 'Sc I', 21.1: 'Sc II',
    22.0: 'Ti I', 22.1: 'Ti II',
    23.0: 'V I',  23.1: 'V II',
    24.0: 'Cr I', 24.1: 'Cr II',
    25.0: 'Mn I', 25.1: 'Mn II',
    26.0: 'Fe I', 26.1: 'Fe II',
    27.0: 'Co I', 27.1: 'Co II',
    28.0: 'Ni I', 28.1: 'Ni II',
    29.0: 'Cu I', 29.1: 'Cu II',
    30.0: 'Zn I',
    38.1: 'Sr II',
    39.0: 'Y I',  39.1: 'Y II',
    40.0: 'Zr I', 40.1: 'Zr II',
    41.0: 'Nb I', 41.1: 'Nb II',
    42.0: 'Mo I',
    44.0: 'Ru I',
    45.0: 'Rh I',
    46.0: 'Pd I',
    47.0: 'Ag I',
    56.0: 'Ba I', 56.1: 'Ba II',
    57.0: 'La I', 57.1: 'La II',
    58.0: 'Ce I', 58.1: 'Ce II',
    59.1: 'Pr II',
    60.0: 'Nd I', 60.1: 'Nd II',
    62.0: 'Sm I', 62.1: 'Sm II',
    63.0: 'Eu I', 63.1: 'Eu II',
    64.0: 'Gd I', 64.1: 'Gd II',
    65.1: 'Tb II',
    66.1: 'Dy II',
    68.0: 'Er I', 68.1: 'Er II',
    69.0: 'Tm I', 69.1: 'Tm II',
    70.0: 'Yb I', 70.1: 'Yb II',
    71.0: 'Lu I',
    72.0: 'Hf I', 72.1: 'Hf II',
    73.0: 'Ta I',
    74.0: 'W I',
    76.0: 'Os I',
    77.0: 'Ir I',
    82.0: 'Pb I',
    90.0: 'Th I', 90.1: 'Th II',
    92.0: 'U I',  92.1: 'U II'}

# -----------------------------------------------------------------------------------------------------------------------
# Elemental Information: Line format: 'Symbol: [Z, Solar Abundance, Solar Abund. Err]
elems = {
    'He': [2.0, 10.93, 0.01], 'Li': [3.0, 1.05, 0.10],
    'C': [6.0, 8.50, 0.06], 'N': [7.0, 7.86, 0.12], 'O': [8.0, 8.76, 0.07],
    'Na': [11.0, 6.24, 0.04], 'Mg': [12.0, 7.60, 0.04],
    'Al': [13.0, 6.45, 0.03], 'Si': [14.0, 7.51, 0.03], 'P': [15.0, 5.41, 0.03], 'S': [16.0, 7.12, 0.03],
    'Cl': [17.0, 5.50, 0.30],
    'K': [19.0, 5.03, 0.09], 'Ca': [20.0, 6.34, 0.04],
    'Sc': [21.0, 3.15, 0.04], 'Ti': [22.0, 4.95, 0.05], 'V': [23.0, 3.93, 0.08], 'Cr': [24.0, 5.64, 0.04],
    'Mn': [25.0, 5.43, 0.04],
    'Fe': [26.0, 7.50, 0.04], 'Co': [27.0, 4.99, 0.07], 'Ni': [28.0, 6.22, 0.04],
    'Cu': [29.0, 4.19, 0.04], 'Zn': [30.0, 4.56, 0.05], 'Sr': [38.0, 2.87, 0.07], 'Y': [39.0, 2.21, 0.05],
    'Zr': [40.0, 2.58, 0.04],
    'Ba': [56.0, 2.18, 0.09], 'Eu': [63.0, 0.52, 0.04], 'Pb': [82.0, 1.75, 0.10]}
