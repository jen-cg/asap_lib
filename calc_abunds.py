import numpy as np
import os
import sys

# ----------------- Import the other files of functions
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from functions.constants import *
from functions.spectra import wavg
from functions.line_list_utils import read_linelist

"""
Calculate Abundances

- Every thing you need to calculate abundances now in one place!
"""


# -----------------------------------------------------------------------------------------------------------------------
# Convert [X/Fe] to A(X)
def xfe_2_AX(xfe, feh, AXx_sol):
    logXH_sol = AXx_sol - 12.

    logXH_star = xfe + feh + logXH_sol

    AX_star = logXH_star + 12.

    return AX_star


# -----------------------------------------------------------------------------------------------------------------------
# Convert A(X) to [X/Fe]
def AX_2_xfe(AX, feh, AXx_sol):
    logXH_sol = AXx_sol - 12.

    logXH_star = AX - 12.

    xfe = logXH_star - logXH_sol - feh

    return xfe


# -----------------------------------------------------------------------------------------------------------------------
def calc_abunds(moog_lines_pars, species, include_uls=False, absolute=False):
    """
    Calculate the abundance of the specified species using a file of moog line parameters

    :param moog_lines_pars: (string) Name of / path to the file containing the moog line parameters
    :param species: (list / tuple)  [atmic_number.ionization state, 'name']  ie [26.0, 'Fe I']
    :param include_uls: (True/False) Include upper limits in the abundance calculation?
    :param absolute: (True/False) Return abundances in absolute form?

    :return: abundance, abundance error, number  of lines used
    """

    # ----------------- Read in the MOOG pars
    with open(moog_lines_pars, 'r') as f:
        lines = f.readlines()

    # ----------------- Extract information from the lines of the  moog_lines_pars file
    atoms = []
    abs_abunds = []
    abund_errs = []
    uls = []
    for line in lines:
        try:
            split = line.split()
            atoms.append(float(split[1]))
            abs_abunds.append(float(split[7]))
            abund_errs.append(float(split[6]))
            uls.append(int(split[10]))
        except:
            pass

    atoms = np.asarray(atoms)
    abs_abunds = np.asarray(abs_abunds)
    abund_errs = np.asarray(abund_errs)
    uls = np.asarray(uls)

    # ----------------- Extract information about only the species of interest
    ax_sol = AX_sol[species[1]]  # Get solar abundance for this species
    good = np.where(atoms == species[0])[0]

    if len(good) > 0:
        abs_abunds = abs_abunds[good]
        abund_errs = abund_errs[good]
        uls = uls[good]
        n_lines = len(abs_abunds)

        # ----------------- Include upper limits in the abundance calculation:
        if include_uls:
            try:
                # Find all upper limits in mlps file
                uls = np.where((uls == 1))[0]  # | (abund_errs == 999.9))[0]
                ul_abs_abunds = abs_abunds[uls]

                # Only take the lowest of the upper limits
                ul_min_abs = np.min(ul_abs_abunds)
            except:
                ul_min_abs = 999.

            # ----------------- Check if non-upper limit lines measure an abundance less than the upper-limit lines
            # If there are real detections with measured values less than the upper limit, we want those instead
            other_mins = np.where(abs_abunds < ul_min_abs)[0]

            if len(other_mins > 0):
                # Turns out there were real detections with measurements lower than the lowest upper limit
                abs_abunds = abs_abunds[other_mins]
                abund_errs = abund_errs[other_mins]

                if len(abs_abunds) > 2:
                    rtn = True
                else:
                    rtn = False

                # Calculate the average abundance and abundance error
                avg_abund, avg_err = wavg(abs_abunds, abund_errs, rtN=rtn)

                xh = avg_abund - ax_sol[0]
                xh_err = np.sqrt(avg_err ** 2 + ax_sol[1] ** 2)
                n_lines = len(abs_abunds)

            else:
                # The lowest upper limit was the lowest value overall
                avg_abund = ul_min_abs
                ax_sol = AX_sol[species[1]]
                xh = avg_abund - ax_sol[0]

                avg_err, xh_err, n_lines = -999, -999, n_lines

        # ----------------- Do not include upper limits in the abundance calculation:
        else:
            # Extract information corresponding to only the non-upper limit lines
            good = np.where(uls == 0)[0]

            if len(good > 0):

                abs_abunds = abs_abunds[good]
                abund_errs = abund_errs[good]

                if len(abs_abunds) > 2:
                    rtn = True
                else:
                    rtn = False

                # Calculate the average abundance and abundance error
                avg_abund, avg_err = wavg(abs_abunds, abund_errs, rtN=rtn)

                xh = avg_abund - ax_sol[0]
                xh_err = np.sqrt(avg_err ** 2 + ax_sol[1] ** 2)
                n_lines = len(abs_abunds)

            else:
                print('No %s lines found in %s' % (species[1], moog_lines_pars))
                avg_abund, avg_err, n_lines = -999, -999, 0
                xh, xh_err, n_lines = -999, -999, 0

    else:
        print('No %s lines found in %s' % (species[1], moog_lines_pars))
        avg_abund, avg_err, n_lines = -999, -999, 0
        xh, xh_err, n_lines = -999, -999, 0

    # ----------------- Return abundance, abundance error, and number of lines used
    if absolute:
        return avg_abund, avg_err, n_lines
    else:
        return xh, xh_err, n_lines


# -----------------------------------------------------------------------------------------------------------------------
def calc_fe(moog_lines_pars, include_uls=False, print_new_Fe=False):
    """
    Calculate the metallicity

    :param moog_lines_pars:  (string) Name of / path to the file containing the moog line parameters
    :param include_uls: (True/False) Include upper limits in the abundance calculation?
    :param print_new_Fe: (True/False) Print

    :return: abundance, abundance error, [total number of lines used, feIh_err, feIIh_err]
    """
    # ----------------- Calculate the metallicity using the Fe I lines
    feIh, feIh_err, n_feI = calc_abunds(moog_lines_pars, [26.0, 'Fe I'], include_uls=include_uls)

    # ----------------- Calculate the metallicity using the Fe II lines
    feIIh, feIIh_err, n_feII = calc_abunds(moog_lines_pars, [26.1, 'Fe II'], include_uls=include_uls)

    mean_fe = -999  # Initialize as -999
    mean_fe_err = -999 # Initialize as -999
    n_fe = -999  # Initialize as -999

    # ----------------- If a non-zero number of lines were used for both FeI and FeII
    if (n_feI > 0) & (n_feII > 0):

        if (feIh_err == -999) & (feIIh_err == -999):
            print('Fe I and II are upper limits!')
            mean_fe = np.mean([feIh, feIIh])
            mean_fe_err = np.std([feIh, feIIh])
            n_fe = [n_feI + n_feII, feIh_err, feIIh_err]

        elif (feIh_err != -999) & (feIIh_err == -999):
            print('Fe II is an upper limit!')
            mean_fe = np.mean([feIh, feIIh])
            fe_std = np.std([feIh, feIIh])
            mean_fe_err = np.sqrt(feIh_err ** 2 + fe_std ** 2)
            n_fe = [n_feI + n_feII, feIh_err, feIIh_err]

        elif (feIh_err == -999) & (feIIh_err != -999):
            print('Fe I is an upper limit!')
            mean_fe = np.mean([feIh, feIIh])
            fe_std = np.std([feIh, feIIh])
            mean_fe_err = np.sqrt(feIIh_err ** 2 + fe_std ** 2)
            n_fe = [n_feI + n_feII, feIh_err, feIIh_err]

        else:
            mean_fe, mean_fe_err = wavg(np.asarray([feIh, feIIh]), np.asarray([feIh_err, feIIh_err]))
            n_fe = [n_feI + n_feII, feIh_err, feIIh_err]

    # ----------------- If no Fe II lines were found
    elif (n_feI > 0) & (n_feII == 0):
        print('No Fe II lines found')
        mean_fe, mean_fe_err, n_fe = feIh, feIh_err, n_feI

    # ----------------- If no Fe I lines were found
    elif (n_feI == 0) & (n_feII > 0):
        print('No Fe I lines found')
        mean_fe, mean_fe_err, n_fe = feIIh, feIIh_err, n_feII

    # ----------------- If no Fe I or II lines were found at all
    else:
        print('Oh no! This really should not happen!')
        print('[Fe I/H] = %s +/- %s (%s lines)' % (round(feIh, 2), round(feIh_err, 2), n_feI))
        print('[Fe II/H] = %s +/- %s (%s lines)' % (round(feIIh, 2), round(feIIh_err, 2), n_feII))
        print('[Fe/H] = %s +/- %s (%s lines)' % (round(mean_fe, 2), round(mean_fe_err, 2), n_fe))

    # ----------------- Print
    if print_new_Fe:
        print('[Fe I/H] = %s +/- %s (%s lines)' % (round(feIh, 2), round(feIh_err, 2), n_feI))
        print('[Fe II/H] = %s +/- %s (%s lines)' % (round(feIIh, 2), round(feIIh_err, 2), n_feII))
        print('[Fe/H] = %s +/- %s (%s lines)' % (round(mean_fe, 2), round(mean_fe_err, 2), n_fe))

    # ----------------- Return mean abundance, abundance error, and number of lines
    return mean_fe, mean_fe_err, n_fe


# -----------------------------------------------------------------------------------------------------------------------
def calc_XFe(other_moog_lines_pars, fe_moog_lines_pars, include_uls=False, print_info=False):
    """
    Calculate the abundance of each element in the provided list with respect to iron

    :param other_moog_lines_pars: (string) Name of / path to the file containing the moog line parameters for the other
    elements (not iron)
    :param fe_moog_lines_pars: (string) Name of / path to the file containing the moog line parameters for iron
    :param include_uls: (True/False)  Include upper limits in the abundance calculation ?
    :param print_info: (True/False) Print information ?

    :return: A nested list of [ X, [X/Fe], error in [X/Fe], number of lines used ]
    """
    # ----------------- Calculate [Fe/H] from a preexisting file
    mean_fe, mean_fe_err, n_fe = calc_fe(fe_moog_lines_pars, include_uls=include_uls, print_new_Fe=print_info)

    # ----------------- Determine what species have been examined
    lwaves, latoms, leps, llgfs = read_linelist(other_moog_lines_pars)
    unique_species = sorted(list(set(latoms)))

    # ----------------- For each unique species
    outs = []
    for u in unique_species:

        # ----------------- If the element is in the dictionary that matches number to species:
        if u in elem_dict.keys():
            # Calculate the abundance
            species = [u, elem_dict[u]]
            mean_x, mean_x_err, n_x = calc_abunds(other_moog_lines_pars, species, include_uls=include_uls)

            xfe = mean_x - mean_fe
            xfe_err = np.sqrt(mean_x_err ** 2 + mean_fe_err ** 2)

            out = [elem_dict[u], round(xfe, 2), round(xfe_err, 2), n_x]
            outs.append(out)

            if print_info:
                print('[%s/Fe] = %s +/- %s (%s lines)' % (elem_dict[u], round(xfe, 2), round(xfe_err, 2), n_x))

        # ----------------- Cannot calculate abundance with this species:
        # JG: TO DO maybe add user input options here
        else:
            print('Missing {} in element dictionary.'.format(u))

    # ----------------- Return
    outs = np.asarray(outs)
    return outs

# -----------------------------------------------------------------------------------------------------------------------
def calc_blends_abunds(final_blends_mlps, include_uls=False, absolute=False):
    # JG: This is the same elem_dict that is in the constants file but without entries for CoI and CoII for unknown reasons
    # elem_dict = {
    #     3.0: 'Li I',
    #     6.0: 'C I', 7.0: 'N I', 8.0: 'O I',
    #     11.0: 'Na I', 12.0: 'Mg I', 13.0: 'Al I',
    #     14.0: 'Si I', 16.0: 'S I',
    #     20.0: 'Ca I', 20.1: 'Ca II',
    #     21.0: 'Sc I', 21.1: 'Sc II',
    #     22.0: 'Ti I', 22.1: 'Ti II',
    #     23.0: 'V I', 23.1: 'V II',
    #     24.0: 'Cr I', 24.1: 'Cr II',
    #     25.0: 'Mn I', 25.1: 'Mn II',
    #     26.0: 'Fe I', 26.1: 'Fe II',
    #     28.0: 'Ni I', 28.1: 'Ni II',
    #     29.0: 'Cu I', 29.1: 'Cu II',
    #     30.0: 'Zn I',
    #     38.1: 'Sr II', 39.1: 'Y II',
    #     40.0: 'Zr I', 40.1: 'Zr II',
    #     56.1: 'Ba II', 57.1: 'La II',
    #     60.1: 'Nd II', 63.1: 'Eu II', 82.0: 'Pb I'}

    # JG: This is the same AX_sol that is in constants.py exept it does not have entries for KI CoI and CoII for unknown reasons
    # AX_sol = {
    #     'Li I': [1.05, 0.10],
    #     'C I': [8.50, 0.06], 'N I': [7.86, 0.12], 'O I': [8.76, 0.07],
    #     'Na I': [6.24, 0.04], 'Mg I': [7.60, 0.04],
    #     'Al I': [6.45, 0.03],
    #     'Si I': [7.51, 0.03], 'S I': [7.12, 0.03],
    #     'Ca I': [6.34, 0.04], 'Ca II': [6.34, 0.04],
    #     'Sc I': [3.15, 0.04], 'Sc II': [3.15, 0.04],
    #     'Ti I': [4.95, 0.05], 'Ti II': [4.95, 0.05],
    #     'V I': [3.93, 0.08], 'V II': [3.93, 0.08],
    #     'Cr I': [5.64, 0.04], 'Cr II': [5.64, 0.04],
    #     'Mn I': [5.43, 0.04], 'Mn II': [5.43, 0.04],
    #     'Fe I': [7.50, 0.04], 'Fe II': [7.50, 0.04],
    #     'Ni I': [6.22, 0.04], 'Ni II': [6.22, 0.04],
    #     'Cu I': [4.19, 0.04], 'Cu II': [4.19, 0.04],
    #     'Zn I': [4.56, 0.05],
    #     'Sr II': [2.87, 0.07], 'Y II': [2.21, 0.07],
    #     'Zr I': [2.58, 0.04], 'Zr II': [2.58, 0.04],
    #     'Ba II': [2.18, 0.09], 'La II': [1.10, 0.04],
    #     'Nd II': [1.42, 0.04], 'Eu II': [0.52, 0.04], 'Pb I': [1.75, 0.10]}

    with open(final_blends_mlps, 'r') as file:
        lines = file.readlines()

    species = []
    waves = []
    abund_errs = []
    abs_abunds = []
    ref_ms = []
    uls = []

    for l in lines:
        split = l.split()
        species.append(round(float(split[1]), 1))
        waves.append(float(split[2]))
        abund_errs.append(round(float(split[5]), 2))
        abs_abunds.append(round(float(split[6]), 2))
        ref_ms.append(round(float(split[7]), 2))
        uls.append(int(split[9]))

    species = np.asarray(species)
    long_name = elem_dict[species[0]]
    ax_sol = AX_sol[long_name]

    abund_errs = np.asarray(abund_errs)
    abs_abunds = np.asarray(abs_abunds)
    ref_ms = np.asarray(ref_ms)
    uls = np.asarray(uls)

    if include_uls:
        try:
            uls = np.where(uls == 1)[0]
            ul_abs_abunds = abs_abunds[uls]
            ul_min_abs = np.min(ul_abs_abunds)
            n_uls = len(ul_abs_abunds)
        except:
            ul_min_abs = 999.

        other_mins = np.where(abs_abunds < ul_min_abs)[0]

        if len(other_mins > 0):
            abs_abunds = abs_abunds[other_mins]
            abund_errs = abund_errs[other_mins]

            if len(abs_abunds) > 2:
                rtn = True
            else:
                rtn = False

            avg_abund, avg_err = wavg(abs_abunds, abund_errs, rtN=rtn)
            xh = avg_abund - ax_sol[0]
            xh_err = np.sqrt(avg_err ** 2 + ax_sol[1] ** 2)
            n_lines = len(abs_abunds)

        else:
            avg_abund = ul_min_abs
            xh = avg_abund - ax_sol[0]

            avg_err, xh_err, n_lines = -999, -999, n_uls

    else:
        good = np.where(uls == 0)[0]

        if len(good > 0):
            abs_abunds = abs_abunds[good]
            abund_errs = abund_errs[good]

            if len(abs_abunds) > 2:
                rtn = True
            else:
                rtn = False

            avg_abund, avg_err = wavg(abs_abunds, abund_errs, rtN=rtn)

            xh = avg_abund - ax_sol[0]
            xh_err = np.sqrt(avg_err ** 2 + ax_sol[1] ** 2)
            n_lines = len(abs_abunds)

        else:
            print('No lines found!')
            avg_abund, avg_err, n_lines = -999, -999, -999
            xh, xh_err, n_lines = -999, -999, -999

    if absolute:

        return avg_abund, avg_err, n_lines

    else:

        return xh, xh_err, n_lines

# -----------------------------------------------------------------------------------------------------------------------
