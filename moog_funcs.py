"""
A bunch of functions that are used by MOOGenstien

The functions below are a bit of nightmare code, I apologize about this.
Most of this was written on the fly with no foresight for future functionality.

There are some paths to places that may need to be changed based on the environment you are working in.
These can be found below the module/package imports.

C.Kielty - 01/2021

Updated on April 23, 2021 - changes made the moog_blends functions to write the ion correctly in output files, plus a
change to some output print statements

"""

import os
import sys
import time
import subprocess
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ----------------- Import the other files of functions
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from functions.handleSpectra import *
from functions.spectra import pyfxcor
from functions.line_list_utils import *



# Command line call for your version of MOOGSILENT. Might just be 'MOOGSILENT'.
# Could also be something like '/usr/local/moognov19/MOOGSILENT'


moog_silent_call = 'MOOGSILENT'  # '/arc5/usr/local/bin/MOOGSILENT'


# -----------------------------------------------------------------------------------------------------------------------
def moogstring2float(string):
    # Convert a string to float...if you can! Mwahahahaha!
    try:
        val = float(string)
    except:
        try:
            val = float(string.replace('D', 'E'))  # JG: I dont know why this is here
        except:
            return None

    return val


# -----------------------------------------------------------------------------------------------------------------------
def gaia_params(pristine_name, gaia_info, semester, solution=1, errs=False):
    """
    I think this is a function to read parameters from different observing semesters
    :param pristine_name:
    :param gaia_info:
    :param semester:
    :param solution:
    :param errs:
    :return:
    """
    star_info = ascii.read(gaia_info)
    colnames = star_info.colnames

    if semester == '2018A':
        names = np.array(star_info['PID'])

        teffs = np.array(star_info['Teff'])
        teffs_errs = np.array(star_info['Teff_unc'])

        loggs = np.array(star_info['logg'])
        loggs_errs = np.array(star_info['logg_unc'])

        mets = np.array(star_info['FeH_spec'])
        phot_mets = np.array(star_info['FeH_phot'])
        mets_errs = np.abs(phot_mets - mets)

        s_names = []
        for n in names:
            s_names.append(n[0] + n[9:12])
        s_names = np.asarray(s_names)

        good = np.where(pristine_name == s_names)[0][0]

        teff = teffs[good]
        teff_err = teffs_errs[good]

        logg = loggs[good]
        logg_err = loggs_errs[good]

        met = mets[good]
        met_err = mets_errs[good]

    if semester == '2018B':
        names = np.array(star_info['name'])

        teffs = np.array(star_info['Teff%s' % solution])
        teffs_errs = np.array(star_info['Teff%s_unc' % solution])

        loggs = np.array(star_info['logg%s' % solution])
        loggs_errs = np.array(star_info['logg%s_unc' % solution])

        mets = np.array(star_info['FERRE_met'])
        phot_mets = np.array(star_info['FeHphot_gi'])
        mets_errs = np.abs(phot_mets - mets)

        good = np.where(pristine_name == names)[0][0]

        teff = teffs[good]
        teff_err = teffs_errs[good]

        logg = loggs[good]
        logg_err = loggs_errs[good]

        met = mets[good]
        met_err = mets_errs[good]

    if semester == '2019A':
        names = np.array(star_info['PID'])

        teffs = np.array(star_info['Teff'])
        teffs_errs = np.array(star_info['Teff_unc'])

        loggs = np.array(star_info['logg'])
        loggs_errs = np.array(star_info['logg_unc'])

        mets = np.array(star_info['met'])
        mets_errs = np.array(star_info['emet'])

        good = np.where(pristine_name == names)[0][0]

        teff = teffs[good]
        teff_err = teffs_errs[good]

        logg = loggs[good]
        logg_err = loggs_errs[good]

        met = mets[good]
        met_err = mets_errs[good]

    if semester == '2019B':
        names = np.array(star_info['name'])

        teffs = np.array(star_info['Teff'])
        teffs_errs = np.array(star_info['Teff_unc'])

        loggs = np.array(star_info['logg'])
        loggs_errs = np.array(star_info['logg_unc'])

        mets = np.array(star_info['met'])
        mets_errs = np.array(star_info['emet'])

        good = np.where(pristine_name == names)[0][0]

        teff = teffs[good]
        teff_err = teffs_errs[good]

        logg = loggs[good]
        logg_err = loggs_errs[good]

        met = mets[good]
        met_err = mets_errs[good]

    params = [teff, logg, met]  # teff_err, logg_err, met_err]

    if errs:
        param_errs = [teff_err, logg_err, met_err]
        return params, param_errs
    else:
        return params


# -----------------------------------------------------------------------------------------------------------------------
def moog_synth_params(name, moog_synth_info, errs=False):
    with open(moog_synth_info, 'r') as file:
        lines = file.readlines()

    nombres = []
    teffs = []
    loggs = []
    mets = []

    teff_errs = []
    logg_errs = []
    met_errs = []

    for l in lines:
        split = l.split()
        nombres.append(split[0])
        teffs.append(float(split[1]))
        loggs.append(float(split[2]))
        mets.append(float(split[3]))

        teff_errs.append(float(split[5]))
        logg_errs.append(float(split[6]))
        met_errs.append(float(split[7]))

    nombres = np.asarray(nombres)
    teffs = np.asarray(teffs)
    loggs = np.asarray(loggs)
    mets = np.asarray(mets)

    teff_errs = np.asarray(teff_errs)
    logg_errs = np.asarray(logg_errs)
    met_errs = np.asarray(met_errs)

    good = np.where(nombres == name)[0][0]

    teff = teffs[good]
    teff_err = teff_errs[good]
    logg = loggs[good]
    logg_err = logg_errs[good]
    met = mets[good]
    met_err = met_errs[good]

    if errs:
        return [teff, logg, met], [teff_err, logg_err, met_err]

    else:
        return [teff, logg, met]


# -----------------------------------------------------------------------------------------------------------------------
def mod_batch(batchpar,
              summary=None,
              smoothed=None,
              spectrum=None,
              model=None,
              linelist=None,
              limits=False,
              wave_range=10.,
              plotpars=False,
              smo=None,
              synth=False,
              nsynth=None,
              atoms=None,
              abunds=None):
    """
    Modify the batch.par file read by moog

    :param batchpar:  name of the batch.par file

    :param summary: (string or None) If not None, the summary entry of the batch.par file will be changed such that the
    moog summary_out file (this is the file path + name for the EW summary or raw synthesis output) is named the same
    as the provided string.

    :param smoothed: (string or None) If not None, the smoothed entry of the batch.par file will be changed such that
    the moog smoothed_out file (this is the file path + name for the smoothed synthetic spectrum output )is named the
    same as the provided string.

    :param spectrum: (string or None) If not None, the spectrum entry of the batch.par file will be changed such that
    the moog observed_in file (this is the file path + name for the input observed spectrum) is named the same as the
    provided string.

    :param model:  (string or None) If not None, the model entry of the batch.par file will be changed such that the
    moog model_in file (this is the file path + name for the inoout model atmosphere) is named the same as the provided
    string.

    :param linelist: (string or None) If not None, the linelist entry of the batch.par file will be changed such that
    the moog lines_in file (this is the file path + name for the input line list) is named the same as the provided
    string.

    :param limits:  (True / False) If True, the synlimits entry of the batch.par file will be changed such that the
    beginning and ending synthesis wavelengths correspond to the minimum and maximum wavelengths from the provided
    linelist file.

    :param wave_range: (float) The additional width in wavelength from the min / max wavelength of the linelist
    for use with plotpars.

    :param plotpars:  (True / False) If True, the plotpars entry of the batch.par file will be changed to contain
    information about the provided linelist file and wave_range.

    :param smo: (float / None) The smoothing factor for use with plotpars

    :param synth: (True / False) If True, the synth entry of the batch.par file will be changed to contain
    information about the provided nsynth, atoms, and abunds information

    :param nsynth: (float / None) If not None, changes the  number of different syntheses to be run by moog, for use
    with synth

    :param atoms: (list or None) If not None, change the abundances in batch.par

    :param abunds: (list or None) If not None, change the abundances in batch.par

    """

    # ----------------- Open the batch.par file
    with open(batchpar, 'r') as file:
        lines = file.readlines()

    # ----------------- Iterate through each line in the file:
    for i, l in enumerate(lines):

        # ----------------- If summary is not none, change the line in lines which contains "summary_out"
        if summary:
            if 'summary_out' in l:
                split = l.split()
                lines[i] = split[0] + '  \'' + summary + '\'\n'

        # ----------------- If model is not none, change the line in lines which contains "model_in"
        if model:
            if 'model_in' in l:
                stop = l.find('\'')
                lines[i] = l[: stop + 1] + model + '\' \n'

        # ----------------- If linelist is not none, change the line in lines which contains "lines_in"
        if linelist:
            if 'lines_in' == l[0:8]:
                stop = l.find('\'')
                lines[i] = l[: stop + 1] + linelist + '\' \n'

        # ----------------- If limits is not none, change the line after the line in lines which contains synlimits
        if limits:
            # Read the linelist
            lwave, latom, lep, llgf = read_linelist(linelist)

            # Find the minimum and maximum wavelength range covered for batch.par
            min_wave = round(np.min(lwave))
            max_wave = round(np.max(lwave))

            limits = [min_wave - wave_range, max_wave + wave_range, 0.01, 1.0]

            if 'synlimits' in l:
                lines[i + 1] = ''
                for s in limits:
                    lines[i + 1] += '  %s' % s
                lines[i + 1] += ' \n'

        # ----------------- If smoothed is not none, change the line in lines which contains "smoothed_out"
        if smoothed:
            if 'smoothed_out' in l:
                stop = l.find('\'')
                lines[i] = l[: stop + 1] + smoothed + '\' \n'

        # ----------------- If spectrum is not none, change the line in lines which contains "observed_in"
        if spectrum:
            if 'observed_in' in l:
                stop = l.find('\'')
                lines[i] = l[: stop + 1] + spectrum + '\' \n'

        # ----------------- If limits is not none, change the line after the line in lines which contains plotpars
        if plotpars:
            ppars = [[min_wave - wave_range, max_wave + wave_range, 0.01, 1.0],
                     [0.0, 0.0, 0.0, 1.0], ['g', smo, 0.0, 0.0, 0.0, 0.0]]
            if 'plotpars' in l:
                for j, p in enumerate(ppars):
                    lines[i + j + 1] = ''
                    for pp in p:
                        lines[i + j + 1] += '  %s' % pp
                    lines[i + j + 1] += ' \n'

        # ----------------- If synth is not none, change the synth entry of the batch.par file
        if synth:
            if 'isotopes' in l:
                lines[i] = l[:-3] + str(nsynth) + ' \n'

            if 'abundances' in l:
                add_lines = []
                end_line = i + 1
                stop = l.find('s')
                lines[i] = l[: stop + 1] + '     %s  %s \n' % (len(atoms), nsynth)
                for j, a in enumerate(atoms):
                    add_line = '  %s' % a
                    for k in range(nsynth):
                        add_line = add_line + '  %s' % abunds[j][k]
                    add_line += ' \n'
                    add_lines.append(add_line)

                add_lines = add_lines[:-1] + [add_lines[-1][:-1]]

                new_lines = lines[:end_line] + add_lines
                lines = new_lines

    # ----------------- Write the new lines to batch.par
    with open(batchpar, 'w') as file:
        file.writelines(lines)


# -----------------------------------------------------------------------------------------------------------------------
def parse_synth_out(smoothed_out, n_synth):
    """
    Parse the output from MOOG Synthesis

    :param smoothed_out: (string) MOOG output file
    :param n_synth: (float) Number of  syntheses that were run by MOOG

    :return: (list: [ np.array([wave1, flux1]), np.array([wave2, flux2]), .... ] ) List of the MOOG smoothed synthetic
    spectra  from each run
    """

    # ----------------- Open the MOOG output file
    with open(smoothed_out, 'r') as file:
        lines = file.readlines()

    # ----------------- Find the locations in the output file corresponding to each synthesis run
    cut_inds = []
    for i, l in enumerate(lines):
        if 'the number ' in l:
            cut_inds.append(i)

    # ----------------- Check the number of syntheses that were run by MOOG
    # In the output file, the output for each synthesis will begin with the text "the number" so the number of times
    # this phrase appears is equal to the number of synthesis runs done by MOOG
    if len(cut_inds) != n_synth:
        print('Number of found synthetic spectra does not match number of syntheses in synth.par')

    # ----------------- Extract the spectrum from each synthesis run
    specs = []
    for i in range(n_synth):
        waves = []
        fluxs = []
        # If there are many runs, then the data occurs between runs minus the two lines evrey run starts with
        if i != n_synth - 1:
            data = lines[cut_inds[i] + 2: cut_inds[i + 1]]
        # If there was only 1 run, then the data starts at the second line
        else:
            data = lines[cut_inds[i] + 2:]
        for j, d in enumerate(data):
            dat = d.split()
            waves.append(float(dat[0]))
            fluxs.append(float(dat[1]))
        spectrum = [waves, fluxs]
        spectrum = np.array(spectrum, dtype=object)
        specs.append(spectrum)

    # ----------------- Save the spectrum from each run
    return specs


# -----------------------------------------------------------------------------------------------------------------------
def run_moog():
    """
    A python wrapper which runs the subprocess moog_silent_call
    """
    # Run MOOGSILENT!
    # mooged = subprocess.call(moog_silent_call)
    mooged = subprocess.call(moog_silent_call, stdout=open(os.devnull, 'wb'))  # To supress output from moog

    if mooged != 0:
        print('MOOG bonked, please check batch.par')


# -----------------------------------------------------------------------------------------------------------------------
# def pyfxcor(obswave, obsflux, synthwave, synthflux, v_tol=5.0, print_vel=False, plot_shift=False):
#     # Make copies of the arrays to not overwrite things
#     t_w = np.copy(obswave)
#     t_f = np.copy(obsflux)
#     t_s_w = np.copy(synthwave)
#     t_s_f = np.copy(synthflux)
#
#     # Interpolate the synthetic spectra on to the observed spectra grid to avoid issues
#     f = interp1d(t_s_w, t_s_f, kind='cubic', bounds_error=False, fill_value=1.0)
#
#     # Transpose the new interpolated flux back onto the original (copied)wavelength grid
#     new_tsf = f(t_w)
#
#     obs = np.copy(t_f)
#     synth = np.copy(new_tsf)
#
#     # Regularize the datasets by subtracting off the mean and dividing by the standard deviation
#     obs -= obs.mean()
#     obs /= obs.std()
#     synth -= synth.mean()
#     synth /= synth.std()
#
#     nsamples = obs.size
#
#     # Find the cross-correlation (this is in pixel values)
#     xcorr = spsi.correlate(obs, synth, method='fft')
#
#     # delta pix array to match xcorr
#     dp = np.arange(1 - nsamples, nsamples)
#     pix_shift = -dp[xcorr.argmax()]
#
#     # Calculate the conversion between pixels and velocity
#     dispersions = []
#     for i in range(len(t_w) - 1):
#         dispersions.append(t_w[i + 1] - t_w[i])
#
#     dispersion = np.mean(dispersions)
#
#     d_lam = dispersion * pix_shift
#
#     lam = np.median(t_s_w)
#
#     vel = s_o_l * (d_lam / lam)
#
#     if print_vel:
#         print(vel)
#
#     # If the corrected velocity is too large,
#     # assume an error occured and make no change
#     if np.abs(vel) > v_tol:
#         corr_wave = np.copy(t_w)
#
#     # If the RV is small, do the shift
#     else:
#         # RV correct the wavelength array
#         corr_wave = np.copy(t_w) * (1.0 + (vel / s_o_l))
#
#     if plot_shift:
#         shifted_f = np.roll(t_f, pix_shift)
#         plt.figure()
#         plt.plot(t_w, new_tsf, 'k', label='Template')
#         plt.plot(t_w, t_f, 'gray', linestyle=':', label='Original Obs')
#         # plt.plot(t_w, shifted_f, label='Corrected Obs1')
#         plt.plot(corr_wave, t_f, label='Corrected Obs')
#         plt.legend()
#         plt.xlabel(r'Wavelength ($\AA$)')
#         plt.ylabel('Normalized Flux')
#         plt.show()
#
#     return corr_wave, vel


# -----------------------------------------------------------------------------------------------------------------------
def parse_moog_out(moog_outs):
    """
    Parse MOOG output files, specifically the MOOG summary_out files.  Usually these will be saved at name.out

    :param moog_outs: (string) Name of or path to the MOOG output file

    :return: (tuple = Absolute abundance, reference metallicity )
    """

    # ----------------- Open the MOOG output file
    with open(moog_outs, 'r') as f:
        olines = f.readlines()

    # ----------------- Extract information from the file
    absolute_abund = []
    ref_metallicity = []
    for i, l in enumerate(olines):
        # Don't waste time reading in the full spectrum
        if i < 10.:
            if 'element' in l:
                absolute_abund = float(l.split()[-1])
            if 'M/H' in l:
                ref_metallicity = float(l.split()[-1].split('=')[-1])

    return absolute_abund, ref_metallicity


# -----------------------------------------------------------------------------------------------------------------------
def determine_uls(obswave,
                  obsflux,
                  linewaves,
                  ref_wave=None,
                  ul_sigma=3.0,
                  line_width=0.5,
                  wave_range=5.0,
                  trim_sigma=[2.0, 2.0],
                  retrim=True):
    """
    This function estimates line depth and determines if a line will be treated as an upper limit or not

    :param obswave: (array) Wavelength array of the spectrum

    :param obsflux: (array) Flux array corresponding to the wavelength array of the spectrum

    :param linewaves: (list/array or float) Wavelength(s) of the spectral lines in question

    :param ref_wave:

    :param ul_sigma: (float)  Upper limit sigma.  If the line depth is greater than ul_sigma times the continuum
    standard deviation then upper_lim = False

    :param line_width: (float) Estimated width of the spectral line in wavelength units (angstroms)

    :param wave_range: (float) In the case of a single line, width of window around the line. In the case of several
    lines, width from the lines with the minimum and maximum wavelengths.  In units of wavelength (angstroms)

    :param trim_sigma: [Upper bound, lower bound].  For use in attempting to remove additional lines when identifying
    the continuum (retrim = True).  Keep only what is within [Upper bound, lower bound] sigma from the mean as the
    continuum.  An appropriate choice of trim_sigma will depend on the signal-to-noise of the spectrum

    :param retrim: (True/False) Attempt to trim additional lines from the continuum ?

    :return: [diff, cont_flux_std, upper_lim]
    - Diff is 1 - the mean continuum flux
    - cont_flux_std is the standard deviation of the continuum flux
    - upper_lim (bool) is 0 if the line depth is greater than the tolerance and is 1 otherwise
    """
    # ----------------  Check if we are fitting one line or many
    # True if many lines (list, tuple, np.ndarray), False if single line (float)
    are_lines = isinstance(linewaves, (list, tuple, np.ndarray))

    # ---------------- Define a window around the line. Large enough to have continuum, small enough to minimize effects
    # of other lines
    if are_lines:
        # If there are many  lines given, take the min and max wavelength of those lines
        short = [np.min(linewaves) - wave_range, np.max(linewaves) + wave_range]
    else:
        # If only one line is given
        short = [linewaves - wave_range, linewaves + wave_range]

    # ---------------- Check the bounds and use the limits of the observed spectrum if needed
    if short[0] < obswave.min():
        short[0] = obswave.min()
    if short[1] > obswave.max():
        short[1] = obswave.max()

    # ---------------- Trim the observed spectrum to the window around the line(s)
    good = np.where((obswave >= short[0]) & (obswave <= short[1]))[0]
    short_wave = obswave[good]
    short_flux = obsflux[good]

    # ---------------- Define a short region of the spectrum +/- line_width from the line center
    if are_lines:
        specline = [ref_wave - line_width, ref_wave + line_width]
    else:
        specline = [linewaves - line_width, linewaves + line_width]

    # ---------------- Check the bounds and use the limits of the trimmed spectrum if needed
    if specline[0] < short_wave.min():
        specline[0] = short_wave.min()
    if specline[1] > short_wave.max():
        specline[1] = short_wave.max()

    # ---------------- Trim the line from the spectrum to find the continuum
    not_line = np.where((short_wave <= specline[0]) | (short_wave >= specline[1]))[0]
    cont_wave = short_wave[not_line]
    cont_flux = short_flux[not_line]

    # ---------------- Attempt to remove other lines
    if retrim:
        cont_flux_mean = np.nanmean(cont_flux)
        cont_flux_std = np.nanstd(cont_flux)
        upper_bound = cont_flux_mean + (trim_sigma[0] * cont_flux_std)
        lower_bound = cont_flux_mean - (trim_sigma[1] * cont_flux_std)
        good = np.where((cont_flux <= upper_bound) & (cont_flux >= lower_bound))[0]
        cont_flux = cont_flux[good]

    # ---------------- Some metrics
    cont_flux_mean = np.nanmean(cont_flux)  # Mean flux value of the continuum
    cont_flux_std = np.nanstd(cont_flux)  # Standard deviation of the flux of the continuum
    diff = 1.0 - cont_flux_mean

    # ---------------- Calculate the depth of the observed line
    # Extract the observed spectrum to the left of the line
    if are_lines:
        left_o_line = np.where(obswave <= ref_wave)[0]
    else:
        left_o_line = np.where(obswave <= linewaves)[0]

    # Extract the index of the spectrum that is closest and to the left of the line
    mindex = np.where(obsflux == obsflux[left_o_line][-1])[0][0]
    # Search around this point for the minumum flux
    line_flux_min = np.min(obsflux[mindex - 2: mindex + 3])
    # Let the line depth be the estimated continuum minus this flux
    line_depth = cont_flux_mean - line_flux_min

    # ---------------- Determine if the line should be treated as an upper limit or not
    ul_tolerance = cont_flux_std * ul_sigma

    if line_depth >= ul_tolerance:
        upper_lim = 0
    else:
        upper_lim = 1

    return [diff, cont_flux_std, upper_lim]


# -----------------------------------------------------------------------------------------------------------------------
def fit_line(abund_info,
             obswave,
             obsflux,
             synthwave,
             synthflux,
             linewaves,
             linedict,
             wave_range,
             diff,
             ref_wave=None,
             rv_corr=True,
             rv_tol=5.0,
             print_rv=False):
    """
    Fit an observed spectral line to a synthetic spectral line.
    That is, determine the depth of the synthetic spectral line, the radial velocity, and the Chi-squared residual
    between the synthetic and observed flux.

    :param abund_info: (list = [a, smog, abs_abund, ref_m] )
    :param obswave: (list)  Wavelengths of the observed spectrum trimmed to the line
    :param obsflux: (list) Fluxes of the observed spectrum corresponding to obswave
    :param synthwave: (list) Wavelengths of the moog synthetic spectrum trimmed to the line
    :param synthflux: (list) Fluxes of the moog synthetic spectrum corresponding to synthwave
    :param linewaves: (float) Wavelength of the line that is being fit
    :param linedict: (dict) The line list you wish to analyze in the dictionary format of func. lines_2_dict
    :param wave_range: (float) pm_line
    :param diff: (float) The difference in the estimated continuum from 1
    :param ref_wave: (bool) 0 if not upper limit, 1 if upper limit
    :param rv_corr: (True/False) If True, correct the radial velocity
    :param rv_tol: (float) Radial velocity tolerance
    :param print_rv: (True / False) If True, print radial velocity

    :return:
    """

    # ----------------- Extract the abundance information
    abund = abund_info[0]
    smo_g = abund_info[1]
    abs_abund = abund_info[2]
    ref_m = abund_info[3]

    # ----------------- Check if we are fitting one line or many
    # True if many lines (list, tuple, np.ndarray), False if single line (float)
    are_lines = isinstance(linewaves, (list, tuple, np.ndarray))

    # ----------------- Define a window around the line +/- wave_range from the line center
    if are_lines:
        # If there are many  lines given, take the min and max wavelength of those lines
        short = [np.min(linewaves) - wave_range, np.max(linewaves) + wave_range]
    else:
        # If only one line is given
        short = [linewaves - wave_range, linewaves + wave_range]

    # ---------------- Check the bounds and use the limits of the synthetic spectrum if needed
    if short[0] < synthwave.min():
        short[0] = synthwave.min()
    if short[1] > synthwave.max():
        short[1] = synthwave.max()

    # ---------------- Trim the observed spectrum to the window around the line(s)
    good_o = np.where((obswave >= short[0]) & (obswave <= short[1]))[0]
    oswave = obswave[good_o]
    osflux = obsflux[good_o]

    # ---------------- Trim the synthetic spectrum to the same window around the line(s)
    good_s = np.where((synthwave >= short[0]) & (synthwave <= short[1]))[0]
    sswave = synthwave[good_s]
    ssflux = synthflux[good_s]

    # ---------------- Extract the synthetic spectrum to the left of the line
    if are_lines:
        left_o_line = np.where(sswave <= ref_wave)[0]
    else:
        left_o_line = np.where(sswave <= linewaves)[0]

    # ---------------- Calculate the depth of the synthetic line
    s_line_flux_min = ssflux[left_o_line][-1]
    sdepth = 1.0 - s_line_flux_min

    # ---------------- If True, RV correct each line
    if rv_corr:
        corr_wave, rv = pyfxcor(oswave, osflux, sswave, ssflux, v_tol=rv_tol, print_vel=print_rv,
                                return_corr_wave=True)
        # Interpolate the radial velocity corrected observed spectrum onto the synthetic grid
        interps = interp1d(corr_wave, osflux, bounds_error=False, fill_value=1.0)
    else:
        # Interpolate the observed spectrum onto the synthetic grid
        interps = interp1d(oswave, osflux, bounds_error=False, fill_value=1.0)
        rv = 0.0

    # ---------------- Interpolated observed flux onto the synthetic grid
    ioflux = interps(sswave)

    # ---------------- Calculate the Chi square residual between the observed and synthetic flux
    r_mean = np.nansum(((ssflux - (ioflux + diff)) ** 2) / ssflux)

    statistic = [abund, abs_abund, ref_m, rv, sdepth, r_mean]

    plot_stuffs = [oswave, osflux, sswave, ssflux, ioflux]

    # ---------------- Add information to the line dictionary
    if are_lines:
        linedict[smo_g] += [statistic]
    else:
        linedict[linewaves][smo_g] += [statistic]

    return linedict, plot_stuffs


# -----------------------------------------------------------------------------------------------------------------------
def parse_mlps(moog_lines_pars, line=None):
    """
    Parse moog line parameters from file

    :param moog_lines_pars: (string) Name of / path to the file containing the moog line parameters
    :param line: (float or None) Wavelength of line to parse information for.  If None, it will return information for
    all lines

    :return: [smoothing parameters, abundance offsets, abundance errors, absolute abundances,
     and reference metallicities]
    """

    # ---------------- Open the moog line parameter file
    with open(moog_lines_pars, 'r') as f:
        lines = f.readlines()

    # ---------------- Extract information from the file
    waves = []
    smogs = []
    abund_offsets = []
    abund_errs = []
    abs_abunds = []
    ref_ms = []

    for l in lines:
        try:
            split = l.split()
            waves.append(float(split[0]))
            smogs.append(float(split[4]))
            abund_offsets.append(float(split[5]))
            abund_errs.append(float(split[6]))
            abs_abunds.append(float(split[7]))
            ref_ms.append(float(split[8]))
        except:
            pass

    waves = np.asarray(waves)
    smogs = np.asarray(smogs)
    abund_offsets = np.asarray(abund_offsets)
    abund_errs = np.asarray(abund_errs)
    abs_abunds = np.asarray(abs_abunds)
    ref_ms = np.asarray(ref_ms)

    # ---------------- If line, extract and return information for only that line
    if line:
        good = np.where(waves == line)[0][0]
        best = [smogs[good], abund_offsets[good], abund_errs[good], abs_abunds[good], ref_ms[good]]

        return best
    # ---------------- Else, extract and return information for all lines
    else:
        return [smogs, abund_offsets, abund_errs, abs_abunds, ref_ms]


# -----------------------------------------------------------------------------------------------------------------------
def moog_looper(spec_name,
                spectrum,
                line_dictionary,
                model_atm,
                smogs=None,
                abundances=None,
                pm_spec=10.,
                pm_line=2.5,
                line_width=0.5,
                retrim=True,
                trim_sigma=[2.0, 2.0],
                ul_sigma=3.0,
                correct_rv=True,
                rv_tolerance=5.0,
                print_prog=False,
                plots=False,
                save_name=None,
                n_x_plots=1):
    """
    A function which loops through different lines and parameters, runs moog on these lines and parameters and compares
    the resulting synthetic spectra to the observed spectra.

    :param spec_name: (string) Name of the object whose spectrum you wish to analyze
    :param spectrum: (string) Path to the spectrum you wish to analyze
    :param line_dictionary: (dict) The line list you wish to analyze in the dictionary format of func. lines_2_dict
    :param model_atm: (string) The path to the model atmosphere

    :param smogs: (list) The size of the Gaussian smoothing kernel in angstroms
    :param abundances: (array) Test abundances
    :param pm_spec: (float) Width with respect to spectral line when trimming
    :param pm_line: (float) In the case of a single line, width of window around the line. In the case of several
    lines, width from the lines with the minimum and maximum wavelengths.  In units of wavelength (angstroms)
    :param line_width: (float) Estimated width of the spectral line in wavelength units (angstroms)
    :param retrim: (True/False) Attempt to trim additional lines from the continuum ?
    :param trim_sigma: [Upper bound, lower bound].  For use in attempting to remove additional lines when identifying
    the continuum (retrim = True).  Keep only what is within [Upper bound, lower bound] sigma from the mean as the
    continuum.  An appropriate choice of trim_sigma will depend on the signal-to-noise of the spectrum
    :param ul_sigma: (float)  Upper limit sigma.  If the line depth is greater than ul_sigma times the continuum
    standard deviation then upper_lim = False
    :param correct_rv: (True/False) If True, correct the radial velocity
    :param rv_tolerance: (float) Radial velocity tolerance
    :param print_prog: (True/False) Print progress ?
    :param plots: (True/False) Display plots?
    :param save_name: (string/None) The name of / path to the file at which to save the plot
    :param n_x_plots:

    :return: The line dictionary updated with fit information
    """

    # ----------------- Define the trimmed spectrum for future use (will be created in current working directory)
    trim_spec = spec_name + '_trimmed.xy'

    # ----------------- Define the MOOG output
    smo_out = spec_name + '.sout'

    # ----------------- Read in the lines from the line_dictionary omitting the "Atmosphere" entry
    keys = line_dictionary.keys()
    lines = []  # wavelength of each line in the line_dictionary
    for k in keys:
        if k != 'Atmosphere':
            lines.append(k)

    n_lines = len(lines)

    # ----------------- Initialize plotting space if desired
    # Initialize a subplot for each of the lines in the line list
    if plots:
        x_plots = n_x_plots
        y_plots = int(round(n_lines / x_plots))

        f, axs = plt.subplots(y_plots, x_plots, figsize=(8, 2 * y_plots))

        if n_lines > 1:
            axs = axs.ravel()
        else:
            axs = np.array([axs], dtype='object')

        for i in range(n_lines, x_plots * y_plots):
            axs[i].axis('off')

        f.tight_layout()

    # ----------------------------------------------Start the big loop-------------------------------------------------
    t0 = time.time()

    # ----------------- For each line in the line list...
    for i, l in enumerate(lines):

        if print_prog:
            print('Starting line {:4d} of {:4d}'.format(i + 1, n_lines))

        # ----------------- Extract the atomic info from the dictionary
        atomic_info = line_dictionary[l]['Atomic Info']

        # ----------------- Define the batch.par values
        added_atoms = [int(round(atomic_info[0]))]  # Get the atomic number

        # ----------------- Create a temporary line list with a single line
        single_line_list(l, atomic_info, outlines='single_line.txt')

        # ----------------- Trim the spectrum to a small window around the single line
        trim_spec_2_linelist(spectrum, 'single_line.txt', pm_spec, trim_spec)

        # ----------------- Read in the trimmed spectrum
        pwave, pflux = read_spec(trim_spec, ftype='xy')

        # ----------------- Determine if the line is an upper limit
        diff, pflux_std, upper_lim = determine_uls(pwave,
                                                   pflux,
                                                   l,
                                                   ref_wave=None,
                                                   ul_sigma=ul_sigma,
                                                   line_width=line_width,
                                                   wave_range=pm_line,
                                                   trim_sigma=trim_sigma,
                                                   retrim=retrim)

        line_dictionary[l]['Upper Limit Info'] = np.array([diff, pflux_std, upper_lim])

        # ----------------- For each smoothing term...
        for smog in smogs:
            smog = round(smog, 2)
            line_dictionary[l][smog] = []

            # ----------------- Iterate through each of the test abundances
            for j, a in enumerate(abundances):
                a = round(a, 2)
                added_abunds = [[a]]

                # ----------------- Modify batch.par
                mod_batch('batch.par',
                          summary=spec_name + '.out',
                          smoothed=smo_out,
                          spectrum=trim_spec,
                          model=model_atm,
                          linelist='single_line.txt',
                          limits=True,
                          wave_range=pm_spec,
                          plotpars=True,
                          smo=smog,
                          synth=True,
                          nsynth=1,
                          atoms=added_atoms,
                          abunds=added_abunds)

                # ----------------- Run moog
                run_moog()

                # ----------------- Parse the MOOG output (extract the MOOG smoothed synthetic spectrum)
                specs = parse_synth_out(smo_out, 1)[0]
                swave = np.asarray(specs[0], dtype=np.float)
                sflux = np.asarray(specs[1], dtype=np.float)

                # ----------------- Estimate how well the observed spectrum matches the synthetic one
                abs_abund, ref_m = parse_moog_out(spec_name + '.out')

                abund_info = [a, smog, abs_abund, ref_m]

                line_dictionary, plot_stuffs = fit_line(abund_info,
                                                        pwave,
                                                        pflux,
                                                        swave,
                                                        sflux,
                                                        l,
                                                        line_dictionary,
                                                        pm_line,
                                                        diff,
                                                        rv_corr=correct_rv,
                                                        rv_tol=rv_tolerance,
                                                        print_rv=False)

                # ----------------- Plot the shifted observed spectrum and the synthetic spectrum at this abundance
                # and smoothing
                if plots:
                    # Observed spectrum shifted
                    axs[i].plot(plot_stuffs[2], plot_stuffs[4] + diff, c='k')

                    # Synthetic spectrum
                    axs[i].plot(plot_stuffs[2], plot_stuffs[3])
                    n_sigma = 1.0 - (ul_sigma * pflux_std)

                    # Add the reference lines
                    axs[i].axhline(1.0, ls='--', c='k')
                    axs[i].axhline(1.0 + pflux_std, ls=':', c='gray')
                    axs[i].axhline(1.0 - pflux_std, ls=':', c='gray')
                    axs[i].axhline(n_sigma, ls=':', c='gray')

        # ----------------- Add a title to the plot
        if plots:
            axs[i].set_title('%s: %s' % (atomic_info[0], l))

    # ----------------- Save plot
    if save_name:
        print('Saving the plot to ' + save_name + '.png')
        f.savefig(save_name + '.png', format='png')

    # ----------------- Print elapsed time
    t1 = time.time()
    if print_prog:
        print('Elapsed time: %s s' % round(t1 - t0))

    return line_dictionary


# -----------------------------------------------------------------------------------------------------------------------
def moog_best_lines(spec_name,
                    spectrum,
                    line_dictionary,
                    model_atm,
                    moog_lines_pars,
                    fe_lines=False,
                    pm_spec=10.,
                    pm_line=2.5,
                    line_width=0.5,
                    retrim=True,
                    trim_sigma=[2.0, 2.0],
                    ul_sigma=3.0,
                    correct_rv=True,
                    rv_tolerance=5.0,
                    print_prog=False,
                    plots=False,
                    save_name=None,
                    n_x_plots=1):
    """
    A function which loops through different lines and parameters, runs moog on these lines and parameters and compares
    the resulting synthetic spectra to the observed spectra.  Unlike moog_looper, this function assumes you already have
    a good idea of the smoothing and metallicity to use.

    :param spec_name: (string) Name of the object whose spectrum you wish to analyze
    :param spectrum: (string) Path to the spectrum you wish to analyze
    :param line_dictionary: (dict) The line list you wish to analyze in dictionary format
    :param model_atm: model_atm: (string) The path to the model atmosphere
    :param moog_lines_pars: (string) Name of / path to the file containing the moog line parameters

    :param fe_lines: (True/False) Does the line dictionary contain Fe lines?
    :param pm_spec: (float) Width with respect to spectral line when trimming
    :param pm_line: (float) In the case of a single line, width of window around the line. In the case of several
    lines, width from the lines with the minimum and maximum wavelengths.  In units of wavelength (angstroms)
    :param line_width: (float) Estimated width of the spectral line in wavelength units (angstroms)
    :param retrim: (True/False) Attempt to trim additional lines from the continuum ?
    :param trim_sigma: [Upper bound, lower bound].  For use in attempting to remove additional lines when identifying
    the continuum (retrim = True).  Keep only what is within [Upper bound, lower bound] sigma from the mean as the
    continuum.  An appropriate choice of trim_sigma will depend on the signal-to-noise of the spectrum
    :param ul_sigma: (float)  Upper limit sigma.  If the line depth is greater than ul_sigma times the continuum
    standard deviation then upper_lim = False
    :param correct_rv: (True/False) If True, correct the radial velocity
    :param rv_tolerance: (float) Radial velocity tolerance
    :param print_prog: (True/False) Print progress ?
    :param plots: (True/False) Display plots?
    :param save_name: (string/None) The name of / path to the file at which to save the plot
    :param n_x_plots:

    :return: The line dictionary updated with fit information
    """

    # ----------------- Define the trimmed spectrum for future use (will be created in current working directory)
    trim_spec = spec_name + '_trimmed.xy'

    # ----------------- Define the MOOG output
    smo_out = spec_name + '.sout'

    # ----------------- Read in the lines from the line_dictionary
    keys = line_dictionary.keys()
    lines = []
    for k in keys:
        if k != 'Atmosphere':
            lines.append(k)

    n_lines = len(lines)

    # ----------------- Initialize plotting space if desired
    if plots:
        x_plots = n_x_plots
        y_plots = int(round(n_lines / x_plots))

        f, axs = plt.subplots(y_plots, x_plots, figsize=(8, 2 * y_plots))

        if n_lines > 1:
            axs = axs.ravel()
        else:
            axs = np.array([axs], dtype='object')

        for i in range(n_lines, x_plots * y_plots):
            axs[i].axis('off')

        f.tight_layout()

    # ----------------------------------------------Start the big loop-------------------------------------------------
    t0 = time.time()

    # ----------------- For each line in the line list...
    for i, l in enumerate(lines):

        if print_prog:
            print('Starting line %s of %s' % (i + 1, n_lines))

        # ----------------- Save the atomic info into the dictionary
        atomic_info = line_dictionary[l]['Atomic Info']

        # ----------------- Define the batch.par values
        added_atoms = [int(round(atomic_info[0]))]

        # ----------------- Create a temporary line list with a single line
        single_line_list(l, atomic_info, outlines='single_line.txt')

        # ----------------- Trim the spectrum to a small window around the single line
        trim_spec_2_linelist(spectrum, 'single_line.txt', pm_spec, trim_spec)

        # ----------------- Read in the trimmed spectrum
        pwave, pflux = read_spec(trim_spec, ftype='xy')

        # ----------------- Determine if the line is an upper limit
        diff, pflux_std, upper_lim = determine_uls(pwave,
                                                   pflux,
                                                   l,
                                                   ref_wave=None,
                                                   ul_sigma=ul_sigma,
                                                   line_width=line_width,
                                                   wave_range=pm_line,
                                                   trim_sigma=trim_sigma,
                                                   retrim=retrim)

        line_dictionary[l]['Upper Limit Info'] = np.array([diff, pflux_std, upper_lim])

        # ----------------- Determine smoothing and abundance parameters
        # Define smogs and abunds based on previous individual fits
        # bests = smo, abund_offset, abund_err, abs_abund, ref_m
        bests = parse_mlps(moog_lines_pars, line=l)
        smogs = [bests[0]]

        # If we are dealing with Fe lines, make sure the abundance offset makes sense
        # based on the Fe of the original and current model atmospheres
        if fe_lines:
            mod_atm_fe = line_dictionary['Atmosphere'][2]
            fe_offset = mod_atm_fe - bests[4]
            new_abund = round(bests[1] - fe_offset, 3)
        else:
            new_abund = bests[1]

        if upper_lim:
            abundances = [new_abund]
            cs = ['b']
            lsty = ['-']
        else:
            new_err = bests[2]
            abundances = [new_abund - new_err, new_abund, new_abund + new_err]
            cs = ['r', 'b', 'r']
            lsty = ['--', '-', '--']

        # ----------------- For each smoothing term:
        for smog in smogs:
            smog = round(smog, 2)
            line_dictionary[l][smog] = []

            # ----------------- For each abundance:
            for j, a in enumerate(abundances):
                a = round(a, 2)
                added_abunds = [[a]]

                # ----------------- Modify batch.par
                mod_batch('batch.par',
                          summary=spec_name + '.out',
                          smoothed=smo_out,
                          spectrum=trim_spec,
                          model=model_atm,
                          linelist='single_line.txt',
                          limits=True,
                          wave_range=pm_spec,
                          plotpars=True,
                          smo=smog,
                          synth=True,
                          nsynth=1,
                          atoms=added_atoms,
                          abunds=added_abunds)

                # ----------------- Run moog
                run_moog()

                # ----------------- Parse the MOOG output (extract the MOOG smoothed synthetic spectrum)
                specs = parse_synth_out(smo_out, 1)[0]
                swave = np.asarray(specs[0], dtype=np.float)
                sflux = np.asarray(specs[1], dtype=np.float)

                # ----------------- Estimate how well the observed spectrum matches the synthetic one
                abs_abund, ref_m = parse_moog_out(spec_name + '.out')

                abund_info = [a, smog, abs_abund, ref_m]

                line_dictionary, plot_stuffs = fit_line(abund_info,
                                                        pwave,
                                                        pflux,
                                                        swave,
                                                        sflux,
                                                        l,
                                                        line_dictionary,
                                                        pm_line,
                                                        diff,
                                                        rv_corr=correct_rv,
                                                        rv_tol=rv_tolerance,
                                                        print_rv=False)

                # ----------------- Plot the shifted observed spectrum and the synthetic spectrum at this abundance
                # and smoothing
                if plots:
                    # Observed spectrum shifted
                    axs[i].plot(plot_stuffs[2], plot_stuffs[4] + diff, c='k')

                    # Synthetic spectrum
                    axs[i].plot(plot_stuffs[2], plot_stuffs[3], label=abs_abund, c=cs[j], ls=lsty[j])

                    # Add the reference lines
                    n_sigma = 1.0 - (ul_sigma * pflux_std)
                    axs[i].axhline(1.0, ls='--', c='k')
                    axs[i].axhline(1.0 + pflux_std, ls=':', c='gray')
                    axs[i].axhline(1.0 - pflux_std, ls=':', c='gray')
                    axs[i].axhline(n_sigma, ls=':', c='gray')

        # ----------------- Add a title to the plot
        if plots:
            axs[i].set_title('%s: %s' % (atomic_info[0], l))
            axs[i].legend(loc='lower right')

    # ----------------- Save plot
    if save_name:
        print('Saving the plot to ' + save_name + '.png')
        f.savefig(save_name + '.png', format='png')

    # ----------------- Print elapsed time
    t1 = time.time()
    if print_prog:
        print('Elapsed time: %s s' % round(t1 - t0))

    return line_dictionary


# -----------------------------------------------------------------------------------------------------------------------
def find_best_abunds(line_dict, ul_sigma=3.0):
    """
    Using a dictionary of line information, find the best fit

    :param line_dict: (dict) A dictionary of line information.  Must include information about synthetic spectra
    :param ul_sigma: (float) If the line is an upper limit, use information about synthetic spectral lines which are
    deeper than ul_sigma times the continuum standard deviation

    :return: (dict) The dictionary of line information updated to include information about the best fit
    """

    # ----------------- Extract relevant sections of the dictionary
    keys = line_dict.keys()
    lines = []
    for k in keys:
        if k != 'Atmosphere':
            lines.append(k)

    # ----------------- For each line in the dictionary:
    for line in lines:

        ul_info = line_dict[line]['Upper Limit Info']
        cont_std = ul_info[1]
        upper_lim = int(ul_info[2])

        skeys = line_dict[line].keys()

        # ----------------- Extract the smoothing parameters
        smogs = []
        for s in skeys:
            try:
                g = float(s)
                smogs.append(g)
            except:
                pass
        smogs = np.asarray(smogs)

        # ----------------- If the line is an upper limit,  find a mean smoothing parameter
        if upper_lim:
            mean_smog = np.mean(smogs)
            s_idx = (np.abs(smogs - mean_smog)).argmin()
            mean_smog = smogs[s_idx]

        # ----------------- For each smoothing parameter:
        bests = []
        for smog in smogs:
            smog = round(smog, 2)

            # ----------------- Iterate though the list of test abundances and extract information:
            abunds = []
            abs_abunds = []
            ref_ms = []
            rvs = []
            sdepths = []
            r_means = []
            for i in range(len(line_dict[line][smog])):
                abunds.append(line_dict[line][smog][i][0])
                abs_abunds.append(line_dict[line][smog][i][1])
                ref_ms.append(line_dict[line][smog][i][2])
                rvs.append(line_dict[line][smog][i][3])
                sdepths.append(line_dict[line][smog][i][4])
                r_means.append(line_dict[line][smog][i][5])

            sdepths = np.asarray(sdepths)
            r_means = np.asarray(r_means)

            # ----------------- Find the set of parameters where the synthetic spectrum best matches the observed one
            # If the line is an upper limit,
            if upper_lim:
                # Calculate where the synthetic line depth is >= some sigma * the continuum error
                ind = np.where(sdepths <= (ul_sigma * cont_std))[0]
                if len(ind) > 0:
                    ind = ind[-1]
                else:
                    print('No accurate upper limit found, increase abundance range')
                    ind = len(sdepths) - 1

            # If the line is not an upper limit,
            else:
                # Calculate the minimum residual and corresponding params
                min_res = np.min(r_means)
                ind = np.where(r_means == min_res)[0][0]

            # Save the best set of parameters
            best_abund = line_dict[line][smog][ind][0]  # Abundance
            best_abs = line_dict[line][smog][ind][1]  # Absolute abundance
            best_refm = line_dict[line][smog][ind][2]  # Reference metallicity
            best_rv = line_dict[line][smog][ind][3]  # Radial velocity
            best_depth = line_dict[line][smog][ind][4]  # Synthetic spectrum depth

            # ----------------- Calculate bounds
            # Bounds based on line depths and continuum uncertainty

            # Get the difference in the synthetic spectrum depths compared to the best synthetic spectrum depth
            sdepth_diffs = np.abs(sdepths - best_depth)
            # Extract synthetic spectra before the best
            lowers = sdepth_diffs[:ind]
            # Extract synthetic spectra after the best
            highers = sdepth_diffs[ind + 1:]

            # Extract synthetic spectra before the best that are deeper than the continuum standard deviation
            lefts = np.where(lowers >= cont_std)[0]
            if lefts.any():
                left = lefts[-1]
                lower_abund = line_dict[line][smog][left][0]
                lower = 1
            else:
                lower = 0

            # Extract synthetic spectra after the best that are deeper than the continuum standard deviation
            rights = np.where(highers >= cont_std)[0]
            if rights.any():
                right = rights[0]
                upper_abund = line_dict[line][smog][ind + 1 + right][0]
                upper = 1
            else:
                upper = 0

            # Calculate the abundance error:
            # If upper and lower == 1
            if lower & upper:
                abund_err = (upper_abund - lower_abund) / 2.
            # If lower == 1 and upper == 0
            elif lower & ~upper:
                abund_err = best_abund - lower_abund
            # If lower == 0 and upper == 1
            elif ~lower & upper:
                abund_err = upper_abund - best_abund
            else:
                abund_err = 999.9

            # Save the best abundance information and abudnace error
            if upper_lim:
                best = [best_abund, round(abund_err, 3), best_abs, best_refm, best_rv]
            else:
                best = [best_abund, round(abund_err, 3), best_abs, best_refm, best_rv, min_res]

            bests.append(best)

        # ---------------- Find where the synthetic spectrum best matches the observed one for all smoothing parameters
        if upper_lim:
            # If the line is an upper limit, take the mean smoothing
            best_val = [mean_smog] + bests[s_idx]
        else:
            # If the line is not an upper limit, take the smoothing where the residual is the smallest
            ress = []
            for b in bests:
                ress.append(b[5])
            ress = np.asarray(ress)
            good = np.where(ress == np.min(ress))[0][0]
            best_val = [smogs[good]] + bests[good][:-1]

        # ---------------- Add the information about the best fit to the dictionary
        line_dict[line]['Best'] = best_val

    return line_dict


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
        wave_len = len(str(k))
        atom_len = len(str(atomic_info[0]))
        ep_len = len(str(atomic_info[1]))
        lgf_len = len(str(atomic_info[2]))

        # Create padding for each element
        w_a_s = ' ' * (10 - atom_len)
        a_e_s = ' ' * (10 - ep_len)
        e_g_s = ' ' * (10 - lgf_len)

        # Write to a string: wavelength, atmoic_number.ionization state, ep, loggf
        string = str(k) + w_a_s + str(atomic_info[0]) + a_e_s + str(atomic_info[1]) + e_g_s + str(atomic_info[2]) \
                 + ' \n'  # The last part ' \n' must be formatted exactly this way or else the moogparams file wont format right

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
        print(new_list)
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
def find_upper_lims(line_dict, sigma, single=False):
    keys = line_dict.keys()

    for k in keys:
        if (k != 'Atmosphere') & (k != 'Atomic Info'):
            smog = k

    abund_array = np.asarray(line_dict[smog])

    comps = []
    for a in abund_array:
        # statistic = [offset(0), abs_abund(1), ref_m(2), rv(3), diff(4), uncert(5),
        #            depth(6), r_mean(7), upper_lim(8)]

        depth = a[6]
        uncert = a[5]

        tolerance = sigma * (uncert / 2.)
        comps.append(tolerance - depth)

    comps = np.asarray(comps)
    good = np.where(comps >= 0.)[0]

    try:
        best_ul = abund_array[good[-1]]

    except:
        print('No accurate upper limit found, increase abundance ratios')
        best_ul = abund_array[-1]

    return best_ul


# -----------------------------------------------------------------------------------------------------------------------
def moog_blend_looper(spec_name, spectrum, line_dictionary, model_atm,
                      smog=None,
                      abundances=None,
                      atom_2_blend=None,
                      blend_list=None,
                      other_elems=None,
                      other_abunds=None,
                      pm_spec=10.,
                      pm_line=2.5,
                      ul_sigma=3.0,
                      line_width=1.0,
                      correct_rv=True,
                      rv_tolerance=5.0,
                      retrim=True,
                      trim_sigma=[2.0, 2.0],
                      print_prog=False,
                      plots=False,
                      fixed_y=None,
                      save_name=None):
    # Define the trimmed spectrum for future use (will be created in current working directory)
    trim_spec = spec_name + '_trimmed.xy'

    # Define the MOOG output
    smo_out = spec_name + '.sout'

    if plots:
        plt.figure()

    ##############################################################################
    # Start the big loop
    t0 = time.time()

    added_atoms = [atom_2_blend]
    if other_elems:
        added_atoms = added_atoms + other_elems

    lwaves, latoms, leps, llgfs = read_linelist(blend_list)

    ref_wave = ref_wave_from_linelist(blend_list)

    line_dictionary['Reference Info'] = np.array([atom_2_blend, ref_wave, blend_list])

    trim_spec_2_linelist(spectrum, blend_list, pm_spec, trim_spec)

    # Read in the trimmed spectrum
    pwave, pflux = read_spec(trim_spec, ftype='xy')

    # Determine if the line is an upper limit
    diff, pflux_std, upper_lim = determine_uls(pwave, pflux, lwaves,
                                               ref_wave=ref_wave,
                                               ul_sigma=ul_sigma,
                                               line_width=line_width,
                                               wave_range=pm_line,
                                               trim_sigma=trim_sigma,
                                               retrim=retrim)

    line_dictionary['Upper Limit Info'] = np.array([diff, pflux_std, upper_lim])

    # For each smoothing term...
    smog = round(smog, 2)
    line_dictionary[smog] = []

    for i, a in enumerate(abundances):
        a = round(a, 2)
        added_abunds = [[a]]

        if other_abunds:
            for o in other_abunds:
                added_abunds = added_abunds + [[o[i]]]

        # Modify batch.par
        mod_batch('batch.par',
                  summary=spec_name + '.out',
                  smoothed=smo_out,
                  spectrum=trim_spec,
                  model=model_atm,
                  linelist=blend_list,
                  limits=True,
                  wave_range=pm_spec,
                  plotpars=True,
                  smo=smog,
                  synth=True,
                  nsynth=1,
                  atoms=added_atoms,
                  abunds=added_abunds)

        ########################################

        # Run moog
        run_moog()

        #######################################

        # Parse the MOOG output
        specs = parse_synth_out(smo_out, 1)[0]
        swave = np.asarray(specs[0], dtype=np.float)
        sflux = np.asarray(specs[1], dtype=np.float)

        ########################################
        # Try to fit the line
        abs_abund, ref_m = parse_moog_out(spec_name + '.out')

        abund_info = [a, smog, abs_abund, ref_m]

        line_dictionary, plot_stuffs = fit_line(abund_info, pwave, pflux,
                                                swave, sflux,
                                                lwaves, line_dictionary, pm_line,
                                                diff,
                                                ref_wave=ref_wave,
                                                rv_corr=correct_rv,
                                                rv_tol=rv_tolerance,
                                                print_rv=False)

        if plots:
            plt.plot(plot_stuffs[2], plot_stuffs[4] + diff, c='k')
            plt.plot(plot_stuffs[2], plot_stuffs[3], label=abs_abund)
            n_sigma = 1.0 - (ul_sigma * pflux_std)
            plt.axhline(1.0, ls='--', c='k')
            plt.axhline(1.0 + pflux_std, ls=':', c='gray')
            plt.axhline(1.0 - pflux_std, ls=':', c='gray')
            plt.axhline(n_sigma, ls=':', c='gray')
            plt.legend(loc='lower right')

    t1 = time.time()

    if print_prog:
        print('Elapsed time: %s s' % round(t1 - t0))

    return line_dictionary


# -----------------------------------------------------------------------------------------------------------------------
def find_best_blend_abunds(line_dict, ul_sigma=3.0):
    keys = line_dict.keys()
    for k in keys:
        try:
            smog = float(k)
        except:
            pass

    smog = round(smog, 2)

    ul_info = line_dict['Upper Limit Info']
    diff = ul_info[0]
    cont_std = ul_info[1]
    upper_lim = int(ul_info[2])

    abunds = []
    abs_abunds = []
    ref_ms = []
    rvs = []
    sdepths = []
    r_means = []

    # statistic = [abund(0), abs_abund(1), ref_m(2), rv(3), sdepth(4), r_mean(5)]
    for i in range(len(line_dict[smog])):
        abund = line_dict[smog][i][0]
        abs_abund = line_dict[smog][i][1]
        ref_m = line_dict[smog][i][2]
        rv = line_dict[smog][i][3]
        sdepth = line_dict[smog][i][4]
        r_mean = line_dict[smog][i][5]

        abunds.append(abund)
        abs_abunds.append(abs_abund)
        ref_ms.append(ref_m)
        rvs.append(rv)
        sdepths.append(sdepth)
        r_means.append(r_mean)

    abunds = np.asarray(abunds)
    abs_abunds = np.asarray(abs_abunds)
    ref_ms = np.asarray(ref_ms)
    rvs = np.asarray(rvs)
    sdepths = np.asarray(sdepths)
    r_means = np.asarray(r_means)

    # Branch if working with upper limits or fits
    if upper_lim:
        # Calculate where the synthetic line depth is >=
        # some sigma * the continuum error
        inds = np.where(sdepths <= (ul_sigma * cont_std))[0]

        if len(inds) > 0:
            ind = inds[-1]
        else:
            print('No accurate upper limit found, change abundance range')
            ind = len(sdepths) - 1

    else:
        # Calculate the minimum residual and corresponding params
        min_res = np.min(r_means)
        ind = np.where(r_means == min_res)[0][0]

    best_abund = line_dict[smog][ind][0]
    best_abs = line_dict[smog][ind][1]
    best_refm = line_dict[smog][ind][2]
    best_rv = line_dict[smog][ind][3]
    best_depth = line_dict[smog][ind][4]

    ####################
    # Bounds based on line depths and continuum uncertainty
    sdepth_diffs = np.abs(sdepths - best_depth)
    lowers = sdepth_diffs[:ind]
    highers = sdepth_diffs[ind + 1:]

    lefts = np.where(lowers >= cont_std)[0]
    if lefts.any():
        left = lefts[-1]
        lower_abund = line_dict[smog][left][0]
        lower = 1

    else:
        lower = 0

    rights = np.where(highers >= cont_std)[0]
    if rights.any():
        upper = 1
        right = rights[0]
        upper_abund = line_dict[smog][ind + 1 + right][0]

    else:
        upper = 0

    if lower & upper:
        abund_err = (upper_abund - lower_abund) / 2.
    elif lower & ~upper:
        abund_err = best_abund - lower_abund
    elif ~lower & upper:
        abund_err = upper_abund - best_abund
    else:
        abund_err = 999.9

    best_val = [smog, best_abund, round(abund_err, 3), best_abs, best_refm, best_rv]

    line_dict['Best'] = best_val

    return line_dict


# -----------------------------------------------------------------------------------------------------------------------
def make_blends_mlps(line_dict, include_uls=False):
    reference_info = line_dict['Reference Info']
    ref_atom = reference_info[0]
    ref_wave = reference_info[1]
    ref_list = reference_info[2]

    split = ref_list.split('_')
    ion = split[2]

    if ion == 'I':
        new_atom = ref_atom + '.0'
    elif ion == 'II':
        new_atom = ref_atom + '.1'

    ref_string = ref_list + '  ' + new_atom + '  ' + ref_wave

    ul_info = line_dict['Upper Limit Info']
    upper_limit = int(ul_info[-1])

    best_info = line_dict['Best']
    # best_info = smog(0), best_abund(1), abund_err(2), best_abs(3), best_refm(4), best_rv(5)

    if include_uls:
        moog_pars = str(best_info[0]) + '  ' + str(best_info[1]) + \
                    '  ' + str(round(best_info[2], 3)) + \
                    '  ' + str(round(best_info[3], 3)) + \
                    '  ' + str(round(best_info[4], 3)) + \
                    '  ' + str(round(best_info[5], 3)) + \
                    '  ' + str(upper_limit)

    else:
        # Select only the lines which are not upper-limit measurements
        if upper_limit == 0:
            moog_pars = str(best_info[0]) + '  ' + str(best_info[1]) + \
                        '  ' + str(round(best_info[2], 3)) + \
                        '  ' + str(round(best_info[3], 3)) + \
                        '  ' + str(round(best_info[4], 3)) + \
                        '  ' + str(round(best_info[5], 3)) + \
                        '  ' + str(upper_limit)

    moog_params = ref_string + '  ' + moog_pars

    return moog_params


# -----------------------------------------------------------------------------------------------------------------------
def moog_best_blend(spec_name, spectrum, line_dictionary, model_atm,
                    best_blend_pars,
                    blend_list=None,
                    other_elems=None,
                    other_abunds=None,
                    pm_spec=10.,
                    pm_line=2.5,
                    ul_sigma=3.0,
                    line_width=1.0,
                    correct_rv=False,
                    rv_tolerance=5.0,
                    retrim=True,
                    trim_sigma=[2.0, 2.0],
                    print_prog=False,
                    plots=False,
                    fixed_y=None,
                    fixed_x=None,
                    save_name=None):
    # Define the trimmed spectrum for future use (will be created in current working directory)
    trim_spec = spec_name + '_trimmed.xy'

    # Define the MOOG output
    smo_out = spec_name + '.sout'

    # best_blend_pars:
    # 'linelist(0)  atom(1)  ref_wave(2)  smog(3)  offset(4)
    # err(5)  abs_abund(6)  ref_m(7)  rv(8)  upperlim(9)'
    split = best_blend_pars.split()

    atom_2_blend = int(float(split[1]))
    ref_wave = float(split[2])
    smog = float(split[3])
    best_abund = round(float(split[4]), 2)
    abund_err = round(float(split[5]), 2)
    upper_lim = int(split[9])

    line_dictionary[smog] = []

    added_atoms = [atom_2_blend]
    if other_elems:
        added_atoms = added_atoms + other_elems

    lwaves, latoms, leps, llgfs = read_linelist(blend_list)

    ref_wave = ref_wave_from_linelist(blend_list)

    line_dictionary['Reference Info'] = np.array([atom_2_blend, ref_wave, blend_list])

    trim_spec_2_linelist(spectrum, blend_list, pm_spec, trim_spec)

    # Read in the trimmed spectrum
    pwave, pflux = read_spec(trim_spec, ftype='xy')

    # Determine if the line is an upper limit
    diff, pflux_std, upper_lim = determine_uls(pwave, pflux, lwaves,
                                               ref_wave=ref_wave,
                                               ul_sigma=ul_sigma,
                                               line_width=line_width,
                                               wave_range=pm_line,
                                               trim_sigma=trim_sigma,
                                               retrim=retrim)

    line_dictionary['Upper Limit Info'] = np.array([diff, pflux_std, upper_lim])

    if upper_lim:
        abundances = [best_abund]
        cs = ['b']
        lsty = ['-']
    else:
        abundances = [best_abund - abund_err, best_abund, best_abund + abund_err]
        cs = ['r', 'b', 'r']
        lsty = ['--', '-', '--']

    # Initialize plotting space if desired
    if plots:
        plt.figure()

    t0 = time.time()

    for i, a in enumerate(abundances):
        added_abunds = [[a]]

        #        print(added_atoms)
        #        print(added_abunds)

        # Modify batch.par
        mod_batch('batch.par',
                  summary=spec_name + '.out',
                  smoothed=smo_out,
                  spectrum=trim_spec,
                  model=model_atm,
                  linelist=blend_list,
                  limits=True,
                  wave_range=pm_spec,
                  plotpars=True,
                  smo=smog,
                  synth=True,
                  nsynth=1,
                  atoms=added_atoms,
                  abunds=added_abunds)

        ########################################

        # Run moog
        run_moog()

        #######################################

        # Parse the MOOG output
        specs = parse_synth_out(smo_out, 1)[0]
        swave = np.asarray(specs[0], dtype=np.float)
        sflux = np.asarray(specs[1], dtype=np.float)

        ########################################
        # Try to fit the line
        abs_abund, ref_m = parse_moog_out(spec_name + '.out')

        abund_info = [a, smog, abs_abund, ref_m]

        line_dictionary, plot_stuffs = fit_line(abund_info, pwave, pflux,
                                                swave, sflux,
                                                lwaves, line_dictionary, pm_line,
                                                diff,
                                                ref_wave=ref_wave,
                                                rv_corr=correct_rv, rv_tol=rv_tolerance,
                                                print_rv=False)

        if plots:
            # plot_stuffs = [oswave, osflux, sswave, ssflux, ioflux]

            if plots:
                # plot_stuffs = oswave[0], osflux[1], sswave[2], ssflux[3], ioflux[4]

                # Observed spectrum shifted
                plt.plot(plot_stuffs[2], plot_stuffs[4] + diff, c='k')

                # Synthetic spectrum
                plt.plot(plot_stuffs[2], plot_stuffs[3], label=abs_abund, c=cs[i], ls=lsty[i])

    if plots:
        plt.title(blend_list)
        plt.axvline(ref_wave, ls=':', c='gray')
        n_sigma = 1.0 - (ul_sigma * pflux_std)
        plt.axhline(1.0, ls='--', c='k')
        plt.axhline(1.0 + pflux_std, ls=':', c='gray')
        plt.axhline(1.0 - pflux_std, ls=':', c='gray')
        plt.axhline(n_sigma, ls=':', c='gray')
        plt.legend(loc='lower right')

        if fixed_x:
            plt.xlim(ref_wave - fixed_x, ref_wave + fixed_x)

        if fixed_y:
            plt.ylim(fixed_y[0], fixed_y[1])

        if save_name:
            plt.savefig(save_name + '.png', format='png')

    t1 = time.time()

    if print_prog:
        print('Elapsed time: %s s' % round(t1 - t0))

    return line_dictionary


