import numpy as np
import os
import sys

# ----------------- Import the other files of functions
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from asap_lib.handleSpectra import read_spec


"""
Line list Utilities 

- Your one stop shop for all your line list needs.  
Read line lists, trim line lists, convert line lists to dictionaries and more 
"""


# -----------------------------------------------------------------------------------------------------------------------
def read_linelist(linelist):
    """
    Read a line list where each line has the format: "wave atom ep lgf"
    - wave: is the wavelength of the transition (angstroms)
    - atom: is the species code, which is atomic_number.ionization_state
    - ep: is the excitation potential
    - lgf: is log(gf) which is a measure of the probability

    :param linelist: (string) Name of / path to the line list
    :return: (arrays) Lists of wavelength, atoms, ep, and lgf
    """

    # ----------------- Read in the line list
    with open(linelist, 'r') as f:
        lines = f.readlines()

    # ----------------- Extract the information from each line
    lwave = []
    latom = []
    lep = []
    llgf = []
    for i, l in enumerate(lines):
        split = l.split()
        try:
            lwave.append(float(split[0]))
            latom.append(float(split[1]))
            lep.append(float(split[2]))
            llgf.append(float(split[3]))
        except:
            pass

    # ----------------- Convert lists to arrays
    lwave, latom, lep, llgf = np.asarray(lwave), np.asarray(latom), np.asarray(lep), np.asarray(llgf)

    return lwave, latom, lep, llgf


# -----------------------------------------------------------------------------------------------------------------------
def lines_2_dict(linelist, stellar_parameters, single=True):
    """
    Convert a line list to a dictionary.

    :param linelist: (string) Name of / path to the line list where each line has the format: "wave atom ep lgf"
    :param stellar_parameters: (list) stellar parameters, must be listed as: [Teff, logg, [Fe/H]]
    :param single: (True/False) Is the dictionary for a single line?
    :return: (dict) The line list as a dictionary

    The dictionary will have keys 'Atmosphere' followed by the wavelengths
    of each line.
    - line_dict['Atmosphere'] = np.array([ *Teff*, *logg*, *metallicity* ]) where Teff, logg, metallicity are the model
    stellar parameters
    - line_dict[*wavelength*] = {'Atomic Info': array([*atomic_number.ionization_state*, *ep*, *lgf* ])}
    eg) line_dict[4871.318] = {'Atomic Info': array([26.   ,  2.865, -0.34 ])}
    """
    # ----------------- Get the line list information
    lwave, latom, lep, llgf = read_linelist(linelist)

    # ----------------- Create a dictionary for each line with the model atmosphere information
    line_dictionary = {'Atmosphere': np.copy(stellar_parameters)}

    # -----------------
    if single:
        # For each line in the line list
        for i, l in enumerate(lwave):
            line_dictionary[l] = {}
            line_dictionary[l]['Atomic Info'] = np.copy([latom[i], lep[i], llgf[i]])
    else:
        line_dictionary['Atomic Info'] = np.copy([lwave, latom, lep, llgf])

    return line_dictionary


# -----------------------------------------------------------------------------------------------------------------------
def dict_2_linelist(dct, spath):
    """
    Create a line list from the entries of a dictionary

    :param dct: The dictionary which contains the line information
    :param spath: Path to save the line list to

    Saves a line list as a txt file
    """

    # ----------------- Remove atmosphere information from the dictionary
    dct.pop('Atmosphere', None)

    # ----------------- Extract line information from the dictionary
    newlines = ['#\n']

    for line in list(dct.keys()):
        l = line
        info = dct[line]['Atomic Info']

        # Create padding for each element
        w_a_s = ' ' * (10 - len(str(info[0])))
        a_e_s = ' ' * (10 - len(str(info[1])))
        e_g_s = ' ' * (10 - len(str(info[2])))

        # Write to a string: wavelength, atmoic_number.ionization state, ep, loggf
        string = str(l) + w_a_s + str(info[0]) + a_e_s + str(info[1]) + e_g_s + str(info[2]) + ' \n'

        newlines.append(string)

    # ----------------- Remove new line character from the last line
    newlines[-1] = newlines[-1].strip('\n')

    # ----------------- Write the final line list
    with open(spath, 'w') as f:
        f.writelines(newlines)


# -----------------------------------------------------------------------------------------------------------------------
def ul_lines_2_dict(linelist, old_linelist, stellar_parameters, single=True):
    lwave, latom, lep, llgf = read_linelist(linelist)
    owave, oatom, oep, olgf = read_linelist(old_linelist)

    new_wave = []
    new_atom = []
    new_ep = []
    new_lgf = []

    for i in range(len(lwave)):

        if latom[i] not in oatom:
            new_wave.append(lwave[i])
            new_atom.append(latom[i])
            new_ep.append(lep[i])
            new_lgf.append(llgf[i])

    new_wave = np.asarray(new_wave)
    new_atom = np.asarray(new_atom)
    new_ep = np.asarray(new_ep)
    new_lgf = np.asarray(new_lgf)

    # Create a dictionary for each line
    line_dictionary = {}

    # Add the model atmosphere information to line_dict
    line_dictionary['Atmosphere'] = np.copy(stellar_parameters)

    if single:

        for i, l in enumerate(new_wave):
            line_dictionary[l] = {}
            line_dictionary[l]['Atomic Info'] = np.copy([new_atom[i], new_ep[i], new_lgf[i]])

    else:
        line_dictionary['Atomic Info'] = np.copy([new_wave, new_atom, new_ep, new_lgf])

    return line_dictionary


# -----------------------------------------------------------------------------------------------------------------------
def single_line_list(line, atomic_information, outlines='single_line.txt'):
    """
    Generate a line list with a single line

    :param line: (float) Wavelength of the line
    :param atomic_information: (array = [atom, ep, lgf]) Atomic information for the line
    :param outlines: (string) Name of / path to the output file
    """
    # ----------------- Extract information from atomic_information
    atom = atomic_information[0]
    ep = atomic_information[1]
    lgf = atomic_information[2]

    # ----------------- Get the number of characters in each item
    atom_len = len(str(atom))
    ep_len = len(str(ep))
    lgf_len = len(str(lgf))

    # ----------------- Make padding between each item
    w_a_s = ' ' * (10 - atom_len)
    a_e_s = ' ' * (10 - ep_len)
    e_g_s = ' ' * (10 - lgf_len)

    # ----------------- Generate final string
    string = ['# \n'] + [str(line) + w_a_s + str(atom) + a_e_s + str(ep) + e_g_s + str(lgf) + ' ']

    # ----------------- Write the final line list
    with open(outlines, 'w') as f:
        f.writelines(string)


# -----------------------------------------------------------------------------------------------------------------------
def trim_spec_2_linelist(spectrum, linelist, plus_minus, outspec):
    """
    Trim a spectrum to a region containing specified line(s) (wavelengths given in linelist) plus a buffer with width
    equal to plus_minus

    :param spectrum: (string) Path to the spectrum (must be in .xy format)
    :param linelist: (string) Path to the line list where each line has the format: "wave atom ep lgf"
    :param plus_minus: (float) Width of the wavelength region around the specified line to trim to in units of array
    indices
    :param outspec: (string) Name of / path to the trimmed spectrum
    """
    # ----------------- Read in the line list
    waves, atoms, eps, lgfs = read_linelist(linelist)

    # ----------------- Get the minimum and maximum wavelengths from the line list
    wave_min = np.min(waves)
    wave_max = np.max(waves)

    # ----------------- Read in the observed spectrum
    with open(spectrum, 'r') as f:
        full = f.readlines()

    # ---------------- For each line in the observed .xy spectrum extract the wavelength and flux
    obswave = []
    obsflux = []
    for i, l in enumerate(full):
        split = l.split()
        obswave.append(float(split[0]))
        obsflux.append(float(split[1]))
    obswave, obsflux = np.asarray(obswave), np.asarray(obsflux)

    # ---------------- Extract the segment of the observed spectrum that is within the min and max wavelengths of the
    # line list plus the width specified by plus_minus
    short_wave = []
    short_flux = []
    for i, w in enumerate(obswave):
        if (w >= wave_min - plus_minus) & (w <= wave_max + plus_minus):
            short_wave.append(w)
            short_flux.append(obsflux[i])

    # ---------------- Format the trimmmed spectrum
    lines = []
    for j in range(len(short_wave)):
        if j != len(short_wave) - 1:
            line = str(short_wave[j]) + ' ' + str(short_flux[j]) + '\n'
            lines.append(line)
        else:
            line = str(short_wave[j]) + ' ' + str(short_flux[j])
            lines.append(line)

    length = len(lines)

    # ---------------- Save the trimmed spectrum
    if length < 10000:
        with open(outspec, 'w') as f:
            f.writelines(lines)
    else:
        print('Trimmed spectrum is too long (%s), please reduce plus_minus to shorten the spectrum' % length)


# -----------------------------------------------------------------------------------------------------------------------
def trim_linelist_2_spectrum(spectrum, line_list, ftype='xy', saveName=None):
    """
    Trim a line list to contain only lines within the wavelength range of the provided spectrum

    :param spectrum: spectrum: (string) Path to the spectrum
    :param line_list:  (string) Path to the line list where each line has the format: "wave atom ep lgf"
    :param ftype: (string) Format of the spectrum file.  Can be 'fits', 'xy', 'bin', or 'cfits'
    :param saveName: (None or string) If None, save the trimmed line list under the original name, or else specify the
    name of the trimmed line list

    """

    # ----------------- Read in the observed spectrum
    obs_w, obs_f = read_spec(spectrum, ftype=ftype)

    # ----------------- Read in the line list
    with open(line_list, 'r') as f:
        lines = f.readlines()

    # ----------------- Extract the information from each line
    trimmed_list = ['#\n']
    for i, l in enumerate(lines):
        split = l.split()
        try:
            lw = float(split[0])
            if np.min(obs_w) <= lw <= np.max(obs_w):
                trimmed_list.append(l)
        except:
            pass

    # ----------------- Remove newline character from the last line of the file
    trimmed_list[-1] = trimmed_list[-1].strip('\n')

    # ----------------- Save updated line list
    if saveName is None:
        with open(line_list, 'w') as f:
            print('Saving trimmed line list to '+line_list)
            f.writelines(trimmed_list)
    else:
        with open(saveName, 'w') as f:
            print('Saving trimmed line list to ' + saveName)
            f.writelines(trimmed_list)


# -----------------------------------------------------------------------------------------------------------------------
def trim_lines_from_list(line_list, remove_lines, saveName=None):
    """

    :param line_list: (string) Path to the list where each line has the format: "wave ....."
    :param remove_lines: (list of floats) List of wavelengths of the lines to remove from the list
    :param saveName: (None or string) If None, save the trimmed line list under the original name, or else specify the
    name of the trimmed line list

    """
    # ----------------- Read in the line list
    with open(line_list, 'r') as f:
        lines = f.readlines()

    # ----------------- Extract the information from each line
    trimmed_list = ['#\n']
    for i, l in enumerate(lines):
        split = l.split()
        try:
            lw = float(split[0])
            if lw not in remove_lines:
                trimmed_list.append(l)
        except:
            pass

    # ----------------- Remove newline character from the last line of the file
    trimmed_list[-1] = trimmed_list[-1].strip('\n')

    # ----------------- Save updated line list
    if saveName is None:
        with open(line_list, 'w') as f:
            print('Saving trimmed list to ' + line_list)
            f.writelines(trimmed_list)
    else:
        with open(saveName, 'w') as f:
            print('Saving trimmed list to ' + saveName)
            f.writelines(trimmed_list)


# -----------------------------------------------------------------------------------------------------------------------
def trim_2_good_list(line_dict, new_list_name, new_moog_params, include_uls=False, all_uls=False):
    """
    Create a list of good lines from a dictionary

    :param line_dict:  (dict) A dictionary of line information.  Must include information about the best fit
    :param new_list_name: (string) Name of / path to the new line list
    :param new_moog_params: (string) Name of / path to the
    :param include_uls: (True/False)
    :param all_uls: (True/False)


    """

    # ----------------- Extract relevant sections of the dictionary
    keys = line_dict.keys()
    lines = []
    for k in keys:
        if k != 'Atmosphere':
            lines.append(k)

    # ----------------- For each line in the dictionary:
    new_list = ['# \n']  # An empty list for the new line list
    moog_params = []  # An empty list for  the moog parameters file
    for i, k in enumerate(lines):

        # Extract information about the line:
        best_info = line_dict[k]['Best']
        ab_err = best_info[2]
        atomic_info = line_dict[k]['Atomic Info']
        ul_info = line_dict[k]['Upper Limit Info']
        upper_limit = int(ul_info[-1])

        # Get the size of each element
        atom_len = len(str(atomic_info[0]))
        ep_len = len(str(atomic_info[1]))
        lgf_len = len(str(atomic_info[2]))

        # Create padding for each element
        w_a_s = ' ' * (10 - atom_len)
        a_e_s = ' ' * (10 - ep_len)
        e_g_s = ' ' * (10 - lgf_len)

        # Write to a string: wavelength, atmoic_number.ionization state, ep, loggf
        string = str(k) + w_a_s + str(atomic_info[0]) + a_e_s + str(atomic_info[1]) + e_g_s + str(atomic_info[2]) \
                 + ' \n'

        # ----------------- Add information to the new lists
        if include_uls:
            if all_uls:
                upper_limit = 1
            moog_pars = str(best_info[0]) + \
                        '  ' + str(best_info[1]) + \
                        '  ' + str(round(best_info[2], 3)) + \
                        '  ' + str(round(best_info[3], 3)) + \
                        '  ' + str(round(best_info[4], 3)) + \
                        '  ' + str(round(best_info[5], 3)) + \
                        '  ' + str(upper_limit) + ' \n'

            moog_params += [string[:-1] + moog_pars]
            new_list += [string]  # Add to the new line list

        else:
            # Select only the lines which are not upper-limit measurements
            if (upper_limit == 0) & (ab_err != 999.9):
                moog_pars = str(best_info[0]) + \
                            '  ' + str(best_info[1]) + \
                            '  ' + str(round(best_info[2], 3)) + \
                            '  ' + str(round(best_info[3], 3)) + \
                            '  ' + str(round(best_info[4], 3)) + \
                            '  ' + str(round(best_info[5], 3)) + \
                            '  ' + str(upper_limit) + ' \n'

                moog_params += [string[:-1] + moog_pars]
                new_list += [string]  # Add to the new line list

    # ----------------- Save information
    if len(moog_params) == 0:
        print(
            'Oh no! No good lines were found in your linelist (maybe all upper limits?) Try setting include_uls=True.')

    else:
        new_list[-1] = new_list[-1][:-1]
        moog_params[-1] = moog_params[-1][:-1]

        # Write the final linelist
        with open(new_list_name, 'w') as f:
            print('Writing file to ' + new_list_name)
            f.writelines(new_list)

        # Write the final MOOG info
        with open(new_moog_params, 'w') as f:
            print('Writing file to ' + new_moog_params)
            f.writelines(moog_params)


# -----------------------------------------------------------------------------------------------------------------------
def ref_wave_from_linelist(linelist):
    split = linelist.split('_')
    ref_wave = float(split[-1].split('.txt')[0])

    return ref_wave


# -----------------------------------------------------------------------------------------------------------------------
def reorder_linelist(line_list, saveName=None):
    """
    Reorder a line list such that the lines are in order of increasing wavelength.
    This might be useful when writing new line lists.

    :param line_list:  (string) Path to the line list where each line has the format: "wave atom ep lgf \n"
    and the first line has the format "# \n"
    :param saveName: (None or string) If None, save the re-ordered line list under the original name, or else specify
    the name of the re-ordered line list

    """

    # ----------------- Read line list
    lst = np.array(open(line_list).readlines())

    # ----------------- Get wavelength of each line in the list
    waves = []
    for line in lst:
        if line != '# \n':
            waves.append(float(line.split()[0]))
    waves = np.array(waves)

    # ----------------- Sort the wavelengths in increasing order
    new = waves.copy()
    new.sort()

    # ----------------- Get the indices of the original list in the new order
    ind = [np.where(waves == n)[0][0] for n in new]

    # ----------------- Reorder the original list (excluding the first line)
    lst = np.array(lst[1:])
    lst = list(lst[ind])

    # ----------------- Make sure each line except the last ends in '\n'
    for i in range(len(lst)):
        if lst[i][-1] != '\n':
            lst[i] += '\n'

    lst[-1] = lst[-1].strip('\n')

    # ----------------- Add back the first line
    lst = ['# \n'] + lst

    # ----------------- Save re-ordered line list
    if saveName is None:
        with open(line_list, 'w') as f:
            print('Saving re-ordered list to ' + line_list)
            f.writelines(lst)
    else:
        with open(saveName, 'w') as f:
            print('Saving re-ordered list to ' + saveName)
            f.writelines(lst)
