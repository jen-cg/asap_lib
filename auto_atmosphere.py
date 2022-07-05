# C. Kielty
# Oct. 2018

"""
Create a MOOG readable model atmosphere from a grid of MARCS models

Any parameters included after
>python auto_atmosphere.py

will be treated as input arguments. PLEASE ONLY USE SPACES TO SEPARATE ARGUMENTS. I.e.:
GOOD:
> python auto_atmosphere.py params.txt atoms.txt modelpath=../atmospheres/ inter_atoms
BAD:
> python auto_atmosphere.py params.txt atoms.txt model path = ../atmospheres/ inter_atoms

Permitted optional arguments:

params.txt: A text file containing Teff, logg, [Fe/H], vmicro, and name
            for stars of interest. Example:
                5501  2.8  -3.5  1.8 Star1
                5989  2.4  -3.1  2.0 Star2
                5889  2.4  -3.1  2.0 Star2_cool
                5770  4.4  0.0
                4289  1.7  -0.5  Arcturus
            Vmicro and name are not required. If vmicro is absent, 1.8 is assumed
            If params.txt is not provided, script will prompt for
            interactive input.

atoms.txt: A text file containing atomic information for the stars in params.txt.
           FIRST LINE OF THE FILE MUST be either 'logFe' or 'absolute' to
           indicate whether the abundances are given as [X/Fe] or A(X)
           Each following line: Z1  Abund1  Z2  Abund2  ...
           If atoms.txt is not provided, script will prompt for interactive
           input. If interactive atomic input is desired, please enter an empty line.
           If no atomic information is desired, use 0.0 for a line.
           If file is provided and inter_atoms is provided, user can add
           elements not in atoms.txt (done for each star individually).
           Example following params.txt example:
               logFe
               6.0 1.2 7.0 0.5 8.0 0.8 12.0 0.4
               6.0 0.8
               14.0 -0.4

               0.0

modelpath=str: FULL path to model atmosphere grids. If not provided, user will
                be prompted. Please finish with a /

outpath=str: FULL path to output location for MOOG formatted atmosphere.
              If out_path is not provided, out_path = current working directory.
              Please finish with a '/'

outtype=params (or) name: The output model atmosphere will either be named based
                           on the stellar parameters or the name provided
                           in params.txt. If params.txt is not provided,
                           but out_name=name then user will be prompted for the
                           name.

inter_grid: Sometimes the script cannot find the right model atmospheres
            to define the interpolation cube. If you would like to interactively
            provide information about which models to use, pass this argument
            to the script.

inter_atoms: Interactively define the atomic abundances for the model?

interactive: Overrides everything and enters full interactive mode

"""
import os
import sys
import subprocess
import operator

# ----------------- Import the other files of functions
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from asap_lib.config import spherical_model_atm_path, plane_parallel_model_atm_path, default_vmicro
from asap_lib.user_input_utils import *
from asap_lib.calc_abunds import *


# -----------------------------------------------------------------------------------------------------------------------
def find_grid_lims(models, print_stuff=True, out=False):
    """
    Find the global dimensions the grid covers(there may be local gaps which cause difficulties)/
    That is, find the grid limits (minimum and maximum values of Teff, log g, and metallicity) for the given set of models

    :param models: (list) the set of models
    :param print_stuff: (True/False) print ranges
    :param out: (True/False). If True, return  [m_teffs, m_loggs, m_mets]
    :return: [m_teffs, m_loggs, m_mets]
    """
    m_teffs = []
    m_loggs = []
    m_mets = []

    # ----------------- Iterate through the models and read the Teff, logg, and metallicity
    for i, m in enumerate(models):
        t_rng = [1, 5]
        m_teffs.append(float(m[t_rng[0]: t_rng[1]]))

        l_rng = [8, 11]
        m_loggs.append(float(m[l_rng[0]: l_rng[1]]))

        start = m.find('z') + 1
        stop = start + 5
        met_rng = [start, stop]
        m_mets.append(float(m[met_rng[0]: met_rng[1]]))

    # ----------------- Print ranges in Teff, logg, and metallicity
    if print_stuff:
        print('Temperature range: %s K - %s K' % (np.min(m_teffs), np.max(m_teffs)))
        print('Gravity range: %s - %s' % (np.min(m_loggs), np.max(m_loggs)))
        print('Metallicity range: %s - %s' % (np.min(m_mets), np.max(m_mets)))

    # ----------------- Return the model Teffs, loggs, and metallicities
    if out:
        return m_teffs, m_loggs, m_mets


# -----------------------------------------------------------------------------------------------------------------------
def find_grid(params, models, bounds):
    """
    Find the models that bound the desired parameters (teff, logg, met) axes of the model cube

    :param params: (list: [teff, logg, met, vmicro, name] ) Stellar parameters
    :param models: (list) The list of model atmospheres
    :param bounds: (list) [t, item, m] where t, item, and m are bools)

    :return: model: (str) The model that most closely matches the parameters and criteria on the bounds
    """

    # ----------------- Find the operations
    ops = []
    for b in bounds:
        # If the item is 0"
        if b == 0:
            # The operation is "less than or equal to"
            ops.append(operator.le)
        # If the item is 1:
        if b == 1:
            # The operation is "greater than or equal to"
            ops.append(operator.ge)

    # ----------------- Set the initial difference thresholds
    dt = 1000
    dl = 6
    dm = 6

    # ----------------- For each model in the set of models:
    for i, m in enumerate(models):

        # ----------------- Get the temperature, logg, and metallicity from the model file name
        t_rng = [1, 5]  # Index range of temperature in file name
        m_t = float(m[t_rng[0]: t_rng[1]])  # Extract the model temperature from the file name
        dt2 = np.abs(m_t - params[0])  # Get the difference in temperature

        l_rng = [8, 11]  # Index range of logg in file name
        m_l = float(m[l_rng[0]: l_rng[1]])  # Extract the model logg from the file name
        dl2 = np.abs(m_l - params[1])  # Get the difference in logg

        start = m.find('z') + 1  # Find the index of "z" in the file name and increase by 1
        stop = start + 5
        met_rng = [start, stop]  # Index range of metallicity in filename
        m_met = float(m[met_rng[0]: met_rng[1]])  # Extract the metallicity from the file name
        dm2 = np.abs(m_met - params[2])  # Get the difference in metallicity

        # ----------------- Check if the model meets the criteria
        # Perform greater than / less than operation for Teff:
        if ops[0](m_t, params[0]):
            # Perform greater than / less than operation for logg:
            if ops[1](m_l, params[1]):
                # Perform greater than / less than operation for metallicity:
                if ops[2](m_met, params[2]):
                    # Check that the differences are below the thresholds:
                    if (dt2 <= dt) & (dl2 <= dl) & (dm2 <= dm):
                        # If all True:
                        model = m
                        # Update the thresholds so that we will search for the closest model atmosphere
                        dt = dt2
                        dl = dl2
                        dm = dm2

    # ----------------- Return model that best meets the criteria
    try:
        return model

    except UnboundLocalError:
        sys.exit(
            'It appears your target parameters are either outside the range of the grid, '
            'or the grid is not well sampled enough to automatically generate the cube. '
            'I recommend either interactive model selection (i_mods = True) or generating a larger grid.')


# -----------------------------------------------------------------------------------------------------------------------
def find_small_grid(parameter, models, param_name, print_stuff=False):
    # Find the models that bound the desired parameter (teff, logg, met) axis of the model cube

    m_ps = []
    for i, m in enumerate(models):

        if param_name == 'teff':
            rng = [1, 5]
        elif param_name == 'logg':
            rng = [8, 11]
        elif param_name == 'met':
            start = m.find('z') + 1
            stop = start + 5
            rng = [start, stop]
        else:
            sys.exit('Unrecognized parameter name. Accepts: teff, logg, met')

        m_ps.append(float(m[rng[0]: rng[1]]))

    mods = np.copy(models)
    mps = np.copy(m_ps)

    # Sort the models by the parameters
    p = zip(mods, mps)
    p.sort(key=lambda x: x[1])
    mods, mps = zip(*p)
    mods, mps = np.asarray(mods), np.asarray(mps)

    if print_stuff:
        print(param_name)
        print(mods)
        print(mps)

    # Check that the desired parameter is contained within the models
    if (parameter >= np.min(mps)) & (parameter <= np.max(mps)):

        if parameter == np.min(mps):
            lowP = np.where(mps <= parameter)[0]
            lowP_mods = mods[lowP]
            lowPs = mps[lowP]

            highP = np.where(mps > parameter)[0]
            highP_mods = mods[highP]
            highPs = mps[highP]

        elif parameter == np.max(mps):
            highP = np.where(mps >= parameter)[0]
            highP_mods = mods[highP]
            highPs = mps[highP]

            lowP = np.where(mps < parameter)[0]
            lowP_mods = mods[lowP]
            lowPs = mps[lowP]

        else:
            highP = np.where(mps >= parameter)[0]
            highP_mods = mods[highP]
            highPs = mps[highP]

            lowP = np.where(mps <= parameter)[0]
            lowP_mods = mods[lowP]
            lowPs = mps[lowP]

    elif parameter > np.max(mps):
        if param_name == 'teff':
            sys.exit(
                'Your grid does not span hot enough temperatures to interpolate around your desired Teff. '
                'Please expand your grid before continuing.')
        elif param_name == 'logg':
            sys.exit(
                'Your grid does not span high enough surface gravities to interpolate around your desired logg. '
                'Please expand your grid before continuing.')
        elif param_name == 'met':
            sys.exit(
                'Your grid does not span high enough metallicities to interpolate around your desired metallicity. '
                'Please expand your grid before continuing.')
        else:
            sys.exit('Unrecognized parameter name. Accepts: teff, logg, met')

    elif parameter < np.min(mps):
        if param_name == 'teff':
            sys.exit(
                'Your grid does not span cool enough temperatures to interpolate around your desired Teff. '
                'Please expand your grid before continuing.')
        elif param_name == 'logg':
            sys.exit(
                'Your grid does not span low enough surface gravities to interpolate around your desired logg. '
                'Please expand your grid before continuing.')
        elif param_name == 'met':
            sys.exit(
                'Your grid does not span low enough metallicities to interpolate around your desired metallicity. '
                'Please expand your grid before continuing.')
        else:
            sys.exit('Unrecognized parameter name. Accepts: teff, logg, met')

    # Check that there are at least the right number of elements in the output arrays
    if param_name == 'teff':
        if len(highP_mods) < 4:
            sys.exit(
                'Your grid does not contain enough hot temperature models to interpolate around your desired Teff. '
                'Please expand your grid before continuing.')
        if len(lowP_mods) < 4:
            sys.exit(
                'Your grid does not contain enough low temperature models to interpolate around your desired Teff. '
                'Please expand your grid before continuing.')

    elif param_name == 'logg':
        if len(highP_mods) < 2:
            sys.exit(
                'Your grid does not contain enough high surface gravity models to interpolate around your desired logg.'
                'Please expand your grid before continuing.')
        if len(lowP_mods) < 2:
            sys.exit(
                'Your grid does not contain enough low surface gravity models to interpolate around your desired logg.'
                ' Please expand your grid before continuing.')

    elif param_name == 'met':
        if len(highP_mods) < 1:
            sys.exit(
                'Your grid does not contain enough high metallicity models to interpolate around your desired'
                ' metallicity. Please expand your grid before continuing.')
        if len(lowP_mods) < 1:
            sys.exit(
                'Your grid does not contain enough low metallicity models to interpolate around your desired '
                'metallicity. Please expand your grid before continuing.')

    else:
        sys.exit('Unrecognized parameter name. Accepts: teff, logg, met')

    return lowP_mods, highP_mods


# -----------------------------------------------------------------------------------------------------------------------
def gen_sub_grids(teff, logg, met, model_num, o_mod):
    lT, hT = find_small_grid(teff, models, 'teff')

    lT_lg, lT_hg = find_small_grid(logg, lT, 'logg')
    hT_lg, hT_hg = find_small_grid(logg, hT, 'logg')

    lT_lg_lm, lT_lg_hm = find_small_grid(met, lT_lg, 'met')
    lT_hg_lm, lT_hg_hm = find_small_grid(met, lT_hg, 'met')
    hT_lg_lm, hT_lg_hm = find_small_grid(met, hT_lg, 'met')
    hT_hg_lm, hT_hg_hm = find_small_grid(met, hT_hg, 'met')

    choice_mods = [lT_lg_lm, lT_lg_hm, lT_hg_lm, lT_hg_hm, hT_lg_lm, hT_lg_hm, hT_hg_lm, hT_hg_hm]

    for i, cm in enumerate(choice_mods):
        m_ps = []
        rng = [1, 5]
        for m in cm:
            m_ps.append(float(m[rng[0]: rng[1]]))

        mods = np.copy(cm)
        mps = np.copy(m_ps)

        # Sort the models by the parameters
        p = zip(mods, mps)
        p.sort(key=lambda x: x[1])
        mods, mps = zip(*p)
        mods, mps = np.asarray(mods), np.asarray(mps)

        choice_mods[i] = mods

    print('Target Params: %s  %s  %s' % (teff, logg, met))
    print('Model atmospheres compatible with model%s (%s): ' % (model_num, mod_opts[model_num - 1]))
    for i, cm in enumerate(choice_mods[model_num - 1]):
        print('%s) %s' % (i + 1, cm))

    err = 1
    again = True
    while again:
        choice = input('Choose ONE model to satisfy %s: (#) ' % mod_opts[model_num - 1])
        try:
            choice = int(choice) - 1
            again = False
        except ValueError:
            print('Sorry, I do not recognize your input, try again ')
            err = err + 1

        if err > 3:
            again = False
            choice = np.where(choice_mods[model_num - 1] == o_mod)[0][0]
            print('Using original model')

    return choice_mods[model_num - 1][choice]


# -----------------------------------------------------------------------------------------------------------------------
def which_grid():
    """
    Decide which grid of MARCS model atmospheres is appropriate based on the log g of the star

    :return: the path to the appropriate grid
    """
    with open('params.txt', 'r') as file:
        lines = file.readlines()

    ref_logg = float(lines[0].split()[1])

    if ref_logg <= 3.5:
        grid = spherical_model_atm_path
        print('-'+grid)

    else:
        grid = plane_parallel_model_atm_path
        print('-'+grid)

    return grid


# -----------------------------------------------------------------------------------------------------------------------
def mod_params_atoms(pristine_name,
                     parameters,
                     plus_minus,
                     T=False,
                     L=False,
                     M=False,
                     V=False,
                     single=False,
                     print_params=False):
    """
    Generate a file with the model atmosphere parameters

    :param pristine_name: name of the star
    :param parameters: stellar parameters [Teff, logg, [Fe/H] ]
    :param plus_minus:
    :param T: True/False
    :param L: True/False
    :param M: True/False
    :param V: True/False
    :param single: True/False
    :param print_params: True/False
    :return:
    """

    # ----------------- Effective Temperature
    teff = parameters[0]
    if teff < 3800.:
        teff = 3800.
    if teff > 7500.:
        teff = 7500.

    # ----------------- Log g
    logg = parameters[1]
    if logg < 0.:
        logg = 0.
    if logg > 5.:
        logg = 5.

    # ----------------- Metallicity
    met = parameters[2]
    if met < -5.:
        met = -5.
    #    if met > -2.0:
    #        met = -2.0

    # ----------------- Microturbulence
    try:
        # If a microturbulence is given do the following:
        vmicro = round(parameters[3], 2)
        if vmicro < 0.0:
            vmicro = 0.1
        if vmicro > 5.0:
            vmicro = 5.0
    except:
        # If a microturbulence is not given, calculate as follows:
        # vmicro = 1.163 + (7.808E-4 * (teff - 5800)) - (0.494 * (logg - 4.3)) - (0.050 * met)
        vmicro = 0.14 - (0.08 * met) + (4.90 * (teff / 10. ** 4)) - (0.47 * logg)
        vmicro = round(vmicro, 2)

    # ----------------- Create the strings for params.txt
    params = []
    string1 = ['%s %s %s %s %s \n' % (teff, round(logg, 2), met, vmicro, pristine_name)]

    # ----------------- If T == True, find the high and low Teff
    if T:
        thi = round(teff + plus_minus)
        tlo = round(teff - plus_minus)

        if thi > 7500.:
            thi = 7500.
        if tlo < 3800.:
            tlo = 3800.

        string2 = ['%s %s %s %s %s \n' % (thi, round(logg, 2),
                                          met, vmicro, pristine_name + '_thi')]
        string3 = ['%s %s %s %s %s' % (tlo, round(logg, 2),
                                       met, vmicro, pristine_name + '_tlo')]

    # ----------------- If L == True, find the high and low Log g
    if L:

        lghi = round(logg + plus_minus, 2)
        lglo = round(logg - plus_minus, 2)

        if logg > 3.5:
            if lghi > 5.0:
                lghi = 5.0
            if lglo < 3.0:
                lglo = 3.0
        else:
            if lghi > 3.5:
                lghi = 3.5
            if lglo < 0.:
                lglo = 0.0

        string2 = ['%s %s %s %s %s \n' % (teff, lghi,
                                          met, vmicro, pristine_name + '_lghi')]
        string3 = ['%s %s %s %s %s' % (teff, lglo,
                                       met, vmicro, pristine_name + '_lglo')]

    # ----------------- If M == True, find the high and low metallicity
    if M:
        mhi = met + plus_minus
        mlo = met - plus_minus

        if mhi > -2.0:
            mhi = -2.0
        if mlo < -5.0:
            mlo = -5.0

        string2 = ['%s %s %s %s %s \n' % (teff, round(logg, 2),
                                          mhi, vmicro, pristine_name + '_mhi')]
        string3 = ['%s %s %s %s %s' % (teff, round(logg, 2),
                                       mlo, vmicro, pristine_name + '_mlo')]

    # ----------------- If V == True, find the high and low microturbulence
    if V:
        vhi = round(vmicro + plus_minus, 2)
        vlo = round(vmicro - plus_minus, 2)

        if vhi > 5.0:
            vhi = 5.0
        if vlo < 0.0:
            vlo = 0.1

        string2 = ['%s %s %s %s %s \n' % (teff, round(logg, 2),
                                          met, vhi, pristine_name + '_vhi')]
        string3 = ['%s %s %s %s %s' % (teff, round(logg, 2),
                                       met, vlo, pristine_name + '_vlo')]

    # ----------------- If single == True,
    if single:
        params += [string1[0][:-2]]
    else:
        params = params + string1 + string2 + string3

    # ----------------- If True, print parameters
    if print_params:
        print('-The model atmosphere parameters are: \n{}'.format(params))

    # ----------------- Write parameters file
    with open('params.txt', 'w') as file:
        file.writelines(params)
        print('-Saving model atmosphere parameters to: params.txt ')

    # ----------------- Write atoms file
    atoms = ['logFe\n']
    if single:
        atoms.append('8.0 0.4 12.0 0.4 14.0 0.4 16.0 0.4 20.0 0.4 22.0 0.4')
    else:
        atoms.append('8.0 0.4 12.0 0.4 14.0 0.4 16.0 0.4 20.0 0.4 22.0 0.4 \n')
        atoms.append('8.0 0.4 12.0 0.4 14.0 0.4 16.0 0.4 20.0 0.4 22.0 0.4 \n')
        atoms.append('8.0 0.4 12.0 0.4 14.0 0.4 16.0 0.4 20.0 0.4 22.0 0.4')

    with open('atoms.txt', 'w') as file:
        file.writelines(atoms)
        print('-Saving atoms to: atoms.txt')


# -----------------------------------------------------------------------------------------------------------------------
def gen_mod_atm(name, params, working_path, print_params=False):
    """
    Generate a MARCS model atmosphere based on the provided stellar parameters

    :param name: Name of the object
    :param params: stellar parameters, must be listed as: [Teff, logg, [Fe/H]]
    :param working_path: working path - saves model atmosphere to this path
    :param print_params: True/False
    :return:
    """
    print('-----------------------------------------------------------------')
    print('----------------- Generating a Model Atmosphere -----------------')
    print('-----------------------------------------------------------------\n')

    # ----------------- Generate a text file with the parameters to use in the model atmosphere
    print('----------------- Finding model atmosphere parameters:')
    mod_params_atoms(name,
                     params,
                     100.,
                     single=True,
                     print_params=print_params)
    print('\n')

    # ----------------- Decide which group of model atmospheres to choose based on the logg coverage
    print('----------------- Finding which group of model atmosphere grids to use based on the log g coverage:')
    grid = which_grid()
    print('\n')

    # ----------------- Run the auto atmosphere
    print('----------------- Creating the model atmospheres:')
    auto_atmosphere(paramstxt=True,
                    atomstxt=True,
                    modelpath=True,
                    model_path=grid,
                    outpath=True,
                    out_path=working_path,
                    out_type='name')

    # ----------------- Cleanup the .alt files (delete them)
    all_files = os.listdir()
    for file in all_files:
        if file.endswith('.alt'):
            os.remove(file)


# -----------------------------------------------------------------------------------------------------------------------
def auto_atmosphere(paramstxt=False,
                    atomstxt=False,
                    modelpath=False,
                    model_path=None,
                    outpath=False,
                    out_path=None,
                    out_type='params',
                    inter_grid=False,
                    inter_atoms=False):
    """
    Generate a model atmosphere

    :param paramstxt: True/False. Is there a file containing stellar parameters for the model atmosphere?
    :param atomstxt: True/False. Is there a file containing atomic information for the model atmosphere?
    :param modelpath: True/False. Do you have the full path to the model atmosphere grids?
    :param model_path: string.  Full path to the model atmosphere grids
    :param outpath: True/False. Do you have the full path to the
    :param out_path: string.  Full path to the model atmosphere grids
    :param out_type: (string, 'params' or 'name') How do you want to name the output model atmosphere?
    :param inter_grid: True/False.
    :param inter_atoms: True/False.
    :return:
    """

    teffs = []
    loggs = []
    mets = []
    vmicros = []
    names = []

    # ------------------------------------------Read in the stellar parameter data-------------------------------------
    # -----------------  If a parameter file exists:
    if paramstxt:
        print('-Reading params.txt')
        with open('params.txt', 'r') as file:
            # read a list of lines into data
            data = file.readlines()

        # ----------------- For each line in params.txt:
        for i, line in enumerate(data):
            stop = False
            item = line.split()

            # ----------------- Teff
            try:
                teffs.append(float(item[0]))
            except IndexError:
                print('\t-No Teff in line %s. Teff will be prompted interactively.' % (i + 1))
                teffs.append('I')
            except ValueError:
                print('\t-Teff in line %s is not a number. Teff will be prompted interactively.' % (i + 1))
                teffs.append('I')

            # ----------------- Log g
            try:
                loggs.append(float(item[1]))
            except IndexError:
                print('\t-No logg in line %s. Logg will be prompted interactively.' % (i + 1))
                loggs.append('I')
            except ValueError:
                print('\t-Logg in line %s is not a number. Logg will be prompted interactively.' % (i + 1))
                loggs.append('I')

            # ----------------- Metallicity
            try:
                mets.append(float(item[2]))
            except IndexError:
                print('\t-No metallicity in line %s. Metallicity will be prompted interactively.' % (i + 1))
                mets.append('I')
            except ValueError:
                print('\t-Metallicity in line %s is not a number. Metallicity will be prompted interactively.' % (
                        i + 1))
                mets.append('I')

            # ----------------- Microturbulent Velocity
            try:
                vmicros.append(float(item[3]))
            except IndexError:
                print('\t-No vmicro in line %s' % (i + 1))
                vmicros.append('D')
                print('\t-No name in line %s' % (i + 1))
                names.append('D')
                stop = True
            except ValueError:
                print('\t-No vmicro in line %s' % (i + 1))
                vmicros.append('D')
                names.append(item[3])
                stop = True

            # ----------------- Name
            if not stop:
                try:
                    names.append(item[4])
                except IndexError:
                    print('\t-No name in line %s' % (i + 1))
                    names.append('D')

    # -----------------  If params.txt is not defined, prompt user for input
    else:
        print('\t-No params.txt provided, starting interactive input for stellar parameters.')
        another = True
        while another:

            # ----------------- Teff
            teffs = int_par_check("Effective Temperature? ", teffs)

            # ----------------- logg
            loggs = int_par_check("Log Surface Gravity? ", loggs)

            # ----------------- Metallicity
            mets = int_par_check("Metallicity? ", mets)

            # ----------------- Microturbulent Velocity
            vmicros.append(input("Microturbulent Velocity? (optional) "))

            # ----------------- Name
            nombre = input("Star Name? (optional) ")
            if nombre.strip():
                noy = no_or_yes(
                    "Would you like to use this name for the output model atmosphere?"
                    " (Default name based on stellar parameters) (N/y) ")
                if noy:
                    names.append('D')
                else:
                    names.append(nombre.strip())
            else:
                names.append('D')

            another = yes_or_no("Would you like to add another star? (Y/n) ")

    # -----------------  For the optional vmicro parameter, replace defaults and blank spaces with the default vmicro
    for i, v in enumerate(vmicros):
        if not v:
            vmicros[i] = default_vmicro
        elif v == 'D':
            vmicros[i] = default_vmicro
        else:
            try:
                vmicros[i] = float(v)
            except ValueError:
                vmicros[i] = default_vmicro

    params = [teffs, loggs, mets, vmicros, names]

    # ------------------------------------------Read in the atomic information data-------------------------------------
    # -----------------  If an atomic information file exists:
    if atomstxt:
        with open('atoms.txt', 'r') as file:
            # read a list of lines into data
            data = file.readlines()

        # Get abundance type from the top of the file
        a_type = data[0].strip()

        # Get abundance data (the remainder of the file)
        all_abunds = data[1:]

        a_names = []
        atoms = []
        abundances = []
        # ----------------- For each line in the atomic information file
        # (Each line represents information for one star)
        for i, line in enumerate(all_abunds):
            a_name = []
            atom = []
            abund = []
            item = line.split()

            # Go through each atom abundance
            for j in range(len(item)):
                try:
                    item[j] = float(item[j])
                except TypeError:
                    continue

                # For every other entry starting at the first entry,
                # this picks out every Z value
                if (j + 1) % 2 == 1:
                    for e in elems:
                        if elems[e][0] == item[j]:
                            a_name.append(e)

                    atom.append(float(item[j]))
                # For every other entry starting at the second entry.
                # this picks out evergy abundance value
                if (j + 1) % 2 == 0:
                    abund.append(float(item[j]))

            a_names.append(a_name)
            atoms.append(atom)
            abundances.append(abund)

        # ----------------- If the abundance type is log Fe, convert the abundance type to absolute units:
        if a_type == 'logFe':
            for i, a in enumerate(atoms):
                for j, n in enumerate(a):
                    AXx_sol = 0
                    found = False
                    for e in elems:
                        if elems[e][0] == n:
                            AXx_sol = elems[e][1]
                            found = True
                    if found:
                        abund = xfe_2_AX(abundances[i][j], mets[i], AXx_sol)
                        abundances[i][j] = round(abund, 2)
                    else:
                        print('\t-Could not find element in my library.')
                        print('\t-Please update the code to include your element.')
            a_type = 'absolute'

    # -----------------If an atomic information file is not given:
    else:
        print('\t-No atoms.txt found. Interactive atomic input will occur later.')
        inter_atoms = True
        a_names = [[] for i in range(len(params[0]))]
        atoms = [[] for i in range(len(params[0]))]
        abundances = [[] for i in range(len(params[0]))]

    # -------------------------------------------- Model and Output Paths ---------------------------------------------
    # ----------------- If the model path was not specified, prompt the user
    if not modelpath:
        print('\t-No model path specified')
        model_path = input('What is the FULL PATH to the directory containing the grid of model atmospheres? ')

    # ----------------- If out_path was not specified, prompt the user
    if not outpath:
        print('\t-No output path specified.')
        opath = yes_or_no("Use current working directory as outpath? (Y/n) ")
        if opath:
            out_path = os.getcwd() + '/'
        else:
            out_path = input('What is the FULL PATH to the directory where you would like output files? ')

    # ----------------- Store the names of all the models to generate model cube later
    all_mods = os.listdir(model_path)

    # ----------------- Make sure we are only looking at .mod files
    models = []
    for m in all_mods:
        if m.endswith('.mod'):
            models.append(m)

    # ----------------- Find grid coverage
    print('-Grid coverage for models in %s:' % model_path)

    find_grid_lims(models)  # This only produces some print statements

    print(
        '(Though your stellar parameters may be within the grid boundaries, local gaps may cause problems. Tread '
        'carefully.)')

    mod_opts = ['Tefflow logglow zlow', 'Tefflow logglow zup', 'Tefflow loggup zlow', 'Tefflow loggup zup',
                'Teffup logglow zlow', 'Teffup logglow zup', 'Teffup loggup zlow', 'Teffup loggup zup']

    # -----------------------------------------------Start the Big Loop ------------------------------------------------
    for i in range(len(params[0])):
        print('\n------- Starting Model Atmosphere Creation for Star %s of %s: ' % (i + 1, len(params[0])))

        teff = teffs[i]
        logg = loggs[i]
        met = mets[i]
        vmicro = vmicros[i]
        name = names[i]

        star_pars = [teff, logg, met, vmicro, name]

        # ----------------- Print parameters
        if name == 'D':
            print('Star Name: Unspecifed')
        else:
            print('Star Name: %s' % name)
        print('Effective Temperature: %s K' % teff)
        print('Surface Gravity: %s' % logg)
        print('Metallicity: %s' % met)
        print('Microturbulent Velocity: %s' % vmicro)

        # -----------------------------------------------Model Atmosphere Cube------------------------------------------
        print('\n------- Generating cube of model stellar atmospheres')
        print('End format needs to be:')

        for j in range(1, 9):
            print('model%s: %s' % (j, mod_opts[j - 1]))

        model_grid = []

        # ----------------- Find the models corresponding to each combination of upper and lower bounds on Teff, logg,
        # and metallicity
        for t in range(0, 2):
            for g in range(0, 2):
                for m in range(0, 2):
                    bounds = [t, g, m]
                    model_grid.append(find_grid(star_pars, models, bounds))
        model_grid = np.array(model_grid)

        print('Cube Found!')
        for j, m in enumerate(model_grid):
            print('model%s: %s' % (j + 1, m))

        # ----------------- User input for model grid selection:
        if inter_grid:
            okay_cube = yes_or_no('Accept this cube of models? (Y/n) ')
            if not okay_cube:
                err = 1
                another = True
                while another:
                    try:
                        change = int(input('Which model would you like to change? (#)'))
                        another = False
                        new = gen_sub_grids(teff, logg, met, change, model_grid[change - 1])
                        model_grid[change - 1] = new
                        for j, m in enumerate(model_grid):
                            print('model%s: %s' % (j + 1, m))

                    except ValueError:
                        print('Sorry, I do not recognize your input, try again ')
                        err = err + 1

                    if err > 3:
                        print('Using original cube')
                        another = False

                    another = yes_or_no('Would you like to change another model? (Y/n) ')

                print('Cube Found!')
                for j, m in enumerate(model_grid):
                    print('model%s: %s' % (j + 1, m))

        # -----------------------------------------------Rewrite Interpol.com------------------------------------------
        print('\n------- Rewriting interpol.com for star %s of %s: ' % (i + 1, len(params[0])))

        # ----------------- Open interpol.com, read the lines
        with open('interpol.com', 'r') as file:
            # read a list of lines into data
            data = file.readlines()

        # ----------------- Update interpol.com
        for d, line in enumerate(data):
            # Change the path to the model atmospheres
            if 'set model_path = ' in line:
                stop = line.find(' = ')
                data[d] = line[:stop + 3] + model_path + '\n'

            # Change the target Teff
            if 'foreach Tref   ( ' in line:
                stop = line.find('( ')
                data[d] = line[:stop + 2] + str(teff) + ' )\n'

            # Change the target logg
            if 'foreach loggref ( ' in line:
                stop = line.find('( ')
                data[d] = line[:stop + 2] + str(logg) + ' )\n'

            # Change the target metallicity
            if 'foreach zref ( ' in line:
                stop = line.find('( ')
                data[d] = line[:stop + 2] + str(met) + ' )\n'

            # Change the output path
            if 'set modele_out2 = ' in line:
                stop1 = line.find(' = ')
                stop2 = line.find('${')
                data[d] = line[:stop1 + 3] + out_path + line[stop2:]
                out_file_name = out_path + line[stop2:]

        # ----------------- Update the models that make the interpolation cube
        k = 1
        while k <= 8:
            for j, line in enumerate(data):
                if ('set model%s = ' % k) in line:
                    stop = line.find(' = ')
                    data[j] = line[:stop + 3] + model_grid[k - 1] + '\n'
                    k += 1

        # ----------------- Write everything back into interpol.com
        with open('interpol.com', 'w') as file:
            file.writelines(data)

        # -----------------------------------------------Run Interpol.com------------------------------------------
        print('\n------- Running interpol.com for star %s of %s: ' % (i + 1, len(params[0])))
        print('\n')
        subprocess.call('./interpol.com')
        # This produces an output file: ${Tref}g${loggref}z${zref}.alt

        # ---------------------------------------Reformat the atmosphere for moog---------------------------------------
        print('\n------- Reformatting output atmosphere for MOOG compatibility for star %s of %s: ' % (
        i + 1, len(params[0])))

        # ----------------- Make the output file names
        ofn = np.copy(out_file_name)
        ofn = str(ofn)

        ofn = ofn.replace('${Tref}', str(teff))
        ofn = ofn.replace('${loggref}', str(logg))
        ofn = ofn.replace('${zref}', str(met))
        ofn = ofn[:-1]
        out_file_alt = ofn

        if out_type == 'params':
            out_file_mmod = ofn[:-3] + 'mmod'
        if out_type == 'name':
            if name == 'D':
                out_file_mmod = ofn[:-3] + 'mmod'
            elif not name:
                out_file_mmod = ofn[:-3] + 'mmod'
            else:
                out_file_mmod = out_path + name + '.mmod'

        # -----------------
        with open(out_file_alt, 'r') as file:
            # read a list of lines into data
            alt_file = file.readlines()

        new_dat = []
        for line in alt_file:
            split = line.split()

            try:
                line_num = int(split[0])

                tau = 10 ** (float(split[1]))
                T = float(split[2])
                logPe = float(split[3])
                logPg = float(split[4])
                rhox = float(split[5])

                new_dat.append('  %6.4e %6.1f  %6.4e  %6.4e \n' % (tau, T, logPg, logPe))

            except:

                if 'INTERPOL' in line:
                    ref_lam = int(float(split[-1]))
                else:
                    continue

        new_dat = ['GENERIC \n', '\n', '                  %s\n' % line_num, '  %s\n' % ref_lam] + new_dat

        # Add vmicro
        new_dat += [' %s\n' % vmicro]

        # --------------------------------------Compile and Update Atmomic Information ---------------------------------
        print('Compiling atomic information for star %s of %s: ' % (i + 1, len(params[0])))

        # Print atomic abundances for this star (remeber "i" iterates through the list of stars)
        if atoms[i]:

            if atoms[i][0] != 0.0:
                print('Atoms loaded for star %s of %s: ' % (i + 1, len(params[0])))

                for j in range(len(atoms[i])):
                    print('%s(%s): %s (%s) ' % (a_names[i][j], atoms[i][j], abundances[i][j], a_type))

                if inter_atoms:
                    inter = yes_or_no("Would you like to add atomic information to the atmosphere model? (Y/n) ")
                else:
                    inter = False

            else:
                print('No atomic information will be added.')
                inter = False
                atoms[i] = []

        else:
            print('Empty atomic entry.')
            inter = yes_or_no("Would you like to add atomic information to the atmosphere model? (Y/n) ")

        # ----------------- Accept user input
        if inter:
            more = True
            err = 1
            while more:
                exid = False
                # Accept an element for input
                elem = input('Which element would you like to add? (atomic number please) ')

                # Check that the input is actually a number
                try:
                    elem = float(elem)
                except ValueError:
                    print('Sorry, I do not recognize that element or your input, try again ')
                    err += 1
                    elem = False

                # Check that the element is real
                if elem > 118.:
                    print('You are studying some exotic matter, please try an element in the periodic table ')
                    err = err + 1
                    elem = False

                # Check the element isn't already loaded:
                if elem in atoms[i]:
                    not_again = no_or_yes('You have already loaded this atom. Would you like to change it? (N/y)')
                    if not_again:
                        elem = False
                    else:
                        exid = True
                        existing_ind = np.where(np.array(atoms[i]) == elem)[0][0]

                # If the above is all good...
                if elem:
                    err = 1

                    # Determine the abundance type
                    convert = fe_or_abs('Is your abundance in units of [X/Fe] or absolute? (Fe, a) ')
                    aerr = 1
                    ask = True

                    while ask:
                        abund = input('What is the abundance? ')
                        # Check that the input is actually a number
                        try:
                            abund = float(abund)
                            ask = False
                        except ValueError:
                            print('Sorry, that was not a number, try again ')
                            aerr += 1
                            if aerr > 3:
                                print('Three strikes, starting again')
                                ask = False

                    if convert:
                        found = False
                        for e in elems:
                            if elems[e][0] == elem:
                                ename = e
                                AXx_sol = elems[e][1]
                                found = True
                        if found:
                            abundance = round(xfe_2_AX(abund, met, AXx_sol), 2)

                            if exid:
                                atoms[i][existing_ind] = elem
                                a_names[i][existing_ind] = ename
                                abundances[i][existing_ind] = abundance

                            else:
                                atoms[i].append(elem)
                                a_names[i].append(ename)
                                abundances[i].append(abundance)

                            print('Input Element: %s (%s)' % (elem, ename))
                            print('[%s/Fe] =  %s' % (ename, abund))
                            print('[Fe/H] = %s' % met)
                            print('A(%s) = %s' % (ename, abundance))
                        else:
                            print('Could not find element in my library.')
                            print('Please try a different element or update the code to include your element.')

                    else:
                        abundance = abund
                        found = False
                        for e in elems:
                            if elems[e][0] == elem:
                                ename = e
                                AXx_sol = elems[e][1]
                                found = True
                        if found:
                            log_ab = round(AX_2_xfe(abund, met, AXx_sol), 2)

                            if exid:
                                atoms[i][existing_ind] = elem
                                a_names[i][existing_ind] = ename
                                abundances[i][existing_ind] = abundance

                            else:
                                atoms[i].append(elem)
                                a_names[i].append(ename)
                                abundances[i].append(abundance)

                            print('Input Element: %s (%s)' % (elem, ename))
                            print('[%s/Fe] =  %s' % (ename, log_ab))
                            print('[Fe/H] = %s' % met)
                            print('A(%s) = %s' % (ename, abundance))
                        else:
                            print('Could not find element in my library')
                            print('Please try a different element or update the code to include your element.')

                if err > 3.:
                    print('Get your shit together!')
                    more = False

                else:
                    more = yes_or_no('Would you like to add another element? (Y/n) ')

        # -----------------------------------Update MOOG Atmosphere with atomic information-----------------------------
        print(
            '\n------- Updating MOOG atmosphere with atomic information for star %s of %s: ' % (i + 1, len(params[0])))

        n_atoms = len(atoms[i])
        new_dat += ['NATOMS        %s          %s\n' % (n_atoms, met)]

        if n_atoms > 0:
            atm_str = '      '
            for j, a in enumerate(atoms[i]):
                atm_str += '%s  %s      ' % (a, round(abundances[i][j], 2))
                if (j + 1) % 3 == 0:
                    atm_str += '\n'
                    new_dat += [atm_str]
                    atm_str = '      '

            if (j + 1) % 3 != 0:
                atm_str += '\n'
                new_dat += [atm_str]

        # -------------------------------------------Add molecular information -----------------------------------------
        print('\n------- Default molecular abundances will be added. Please check your atmosphere model to make sure'
              ' all is okay.')

        new_dat += ['NMOL          16\n']
        new_dat += ['       1.1     107.0     108.0     607.0     608.0     708.0       6.1\n']
        new_dat += ['       7.1       8.1      12.1     112.0     101.0     106.0     101.0\n']
        new_dat += ['      22.1     822.0\n']

        # -----------------------------------------------Write the final file ------------------------------------------
        print('\n------- Writing the final MOOG compatible atmosphere for star {} of {}'.format(i + 1, len(params[0])))

        with open(out_file_mmod, 'w') as file:
            file.writelines(new_dat)
            print('MOOG compatible atmosphere saved to ' + out_file_mmod)
