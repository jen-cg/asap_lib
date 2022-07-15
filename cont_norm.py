import os
import sys
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from scipy.ndimage import median_filter
import scipy.stats as spst
import scipy.signal as spsi
from scipy.interpolate import interp1d

# ----------------- Import the other files of functions
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from asap_lib.handleSpectra import *
from asap_lib.spectra import order_chop, find_order_splits


def contnorm_sigclip(y, sigma_lower, sigma_upper):
    """
    Improve the continuum position with asymetric sigma-clipping
    ------------
    Best For: Spectra which have already been continuum normalized such that
    the continuum is already horizontal. This routine will bring the continuum
    value closer to 1.

    - New continuum normalization code (written by J. Glover)

    :param y: (list or array) Flux values
    :param sigma_lower: (float) Lower sigma value for clipping
    :param sigma_upper: (float) Upper sigma value for clipping

    Returns: normalized flux array
    """

    # Sigma-clip to estimate the continuum value
    sol = sigma_clip(y, sigma_lower=sigma_lower, sigma_upper=sigma_upper, maxiters=None)
    clip_norm = [y[i] for i in range(len(sol.mask)) if sol.mask[i] == False]

    # Divide flux by the average estimated continuum value to bring
    # the continuum closer to 1
    norm = y / np.mean(clip_norm)

    return norm


# -----------------------------------------------------------------------------------------------------------------------
def contnorm_filter(y, smooth_kernel, mode='reflect'):
    """
    Continuum Normalization by estimating the continuum with a filter
    ------------
    Best For: "Raw" spectra which have not been continuum normalized yet.

    - New continuum normalization code (written by J. Glover)

    :param y: (list or array) Flux values
    :param smooth_kernel: (int) Length of the kernel in units of array indices (pix)
    :param mode: (float) Descirbes how the input array is extended beyond its boundaries

    returns: normalized flux array, filtered flux array
    """

    filt = median_filter(y, size=smooth_kernel, mode=mode)

    # To avoid divide by zero errors, wherever the filter = 0, set the filter to 1
    ind = np.where(filt == 0)
    filt[ind] = 1

    norm = y / filt

    return norm, filt


# -----------------------------------------------------------------------------------------------------------------------
def contnorm_2stage(y, smooth_kernel, sigma_lower, sigma_upper, mode='reflect'):
    """
    Two-Stage Continuum Normalization Routine
    ------------
    Best For: "Raw" spectra which have not been continuum normalized yet.

    - New continuum normalization code (written by J. Glover)

    Stage 1: Median Filter Normalization

    Find the first estimate of the continuum by applying a
    median filter with width = smooth_kernel to the spectrum.
    The spectrum is normalized by dividing by the filtered spectrum.
        - Your results will be affected by the size of smooth_kernel.
        A kernel with a width that is a significant fraction of the total length of the spectrum
        is a good place to start ( ie smooth_kernel = len(y)/2 )

    Stage 2: k-sigma clipping Normalization

    Find the second estimate of the continuum by applying a k-sigma clipping routine
    to the normalized spectrum from stage 1. Normalize the spectrum a second time by dividing by the
    mean of the continuum.
        - Your results will be affected by the choice of the upper and lower sigma limits.
        simga_lower = 1.5 and sigma_upper = 3 is good place to start

    Returns
    -----
    norm2: (array) The continuum-normalized spectrum
    filt: (array) The filtered spectrum. Useful for checking and adjusting the smooth_kernel size
    """

    # ------- Step 1: Smooth the flux to get the first estimate of the continuum, then norm
    norm, filt = contnorm_filter(y, smooth_kernel, mode=mode)

    # ------ Step 2: Improve the estimate of the continuum. Sigma clip the normalized flux. Then norm again
    norm2 = contnorm_sigclip(norm, sigma_lower, sigma_upper)

    return norm2, filt


# -----------------------------------------------------------------------------------------------------------------------
class drawContinuum:
    """
    Estimate the continuum function via user input and perform continuum normalization
    ---------------
    Connects to a matplotlib figure via matplotlib event handling and allows user to draw a continuum by placing
     points on the given figure.
    - Press 1 to place points, press 0 to finish, press any other key to pause.

    Best For: "Raw" spectra which have not been continuum normalized yet, especially those with difficult and/or
    unusual continuums. For example, spectra which contain deep and wide spectral lines.

    - New continuum normalization code (written by J. Glover)

    Parameters
    ----
    :param fig: Matplotlib Figure instance
    :param ax: Matplotlib Axes instance
    :param w: (list or array) wavelength array
    :param f: (list or array) flux array corresponding to w

    Attributes
    ----
    self.cont: list of continuum values (corresponds to same wavelength grid as w)
    self.norm: list of normalized flux (corresponds to same wavelength grid as w)
    self.xs: list of x positions of user click locations
    self.ys: list of y positions of user click locations


    Example Useage
    ----
    %matplotlib notebook
    import matplotlib.pyplot as plt
    import numpy as np
    import asap_lib.spectra as sa
    import asap_lib.cont_norm as cn

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.set_title('Click to place points along the continuum\n Press: 1 to place points, 0 to finish,
     any other key to pause')

    # ---------- Plot original spectrum
    plt.plot(w,f)

    # ---------- Initialize and create continuum object
    continuum = cn.drawContinuum(fig, ax, w, f)
    continuum.connect()

    #
    #  ... User places points along the continuum ...
    #

    # ---------- Normalize
    continuum.disconnect()
    continuum.normalize()

    # ---------- Save the continuum normalized spectrum
    sa.write2bin(w, continuum.norm, err, save_path )

    """

    def __init__(self, fig, ax, w, f):
        self.key = ['1']
        self.done = [0]

        self.fig = fig
        self.ax = ax

        self.w = w
        self.f = f

        self.xs = []
        self.ys = []

        self.cont = []
        self.norm = []

        # ----- Initialize the legend
        blank = plt.scatter([], [])
        leg = self.ax.legend([blank], ['On'], handlelength=0, handletextpad=0, fancybox=True, loc='upper right')
        for item in leg.legendHandles:
            item.set_visible(False)

    def connect(self):
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_press = self.fig.canvas.mpl_connect('key_press_event', self.onpress)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.fig.canvas.mpl_disconnect(self.cid_press)

    # -----  Define the events
    def onclick(self, event):

        # If the last keystroke was a 0, quit
        if self.key[-1] == '0':
            self.done.append(1)
            self.fig.canvas.mpl_disconnect(self.cid_click)

        # If the last keystroke was a 1, place a scatter point on the cursor location
        if self.key[-1] == '1':
            self.ax.scatter(event.xdata, event.ydata, color='tab:red')
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)

        # If the last keystroke was anything else, do not place anymore points until the user hits 1 again

    def onpress(self, event):
        self.key.append(event.key)

        blank = plt.scatter([], [])

        if self.key[-1] == '0':
            leg = self.ax.legend([blank], ['Done'], handlelength=0, handletextpad=0, fancybox=True, loc='upper right')

        elif self.key[-1] == '1' and self.done[-1] != 1:
            leg = self.ax.legend([blank], ['On'], handlelength=0, handletextpad=0, fancybox=True, loc='upper right')

        elif self.done[-1] != 1:
            leg = self.ax.legend([blank], ['Paused'], handlelength=0, handletextpad=0, fancybox=True, loc='upper right')

        for item in leg.legendHandles:
            item.set_visible(False)

    def normalize(self):
        self.cont = np.interp(self.w, self.xs, self.ys)
        self.norm = self.f / self.cont


# -----------------------------------------------------------------------------------------------------------------------
def sig_clip(wave, flux, clo, chi, window, step, reference):
    """
    Original Continuum Position Adjustment via sigma-clipping (written by C. Kielty)
    -------------------
    Perform sigma-clipping on the provided spectrum or spectrum segment

    Best For: Standardize OPERA

    Inputs
    ------

    wave: the wavelength array

    flux: the flux array corresponding to the wavelength array

    clo: the lower sigma-cipping limit

    chi: the upper sigma-clipping limit

    window: Size of the sliding boxar in pixels; needs to be an odd-number

    step: some value that the Savitzky-Golay filter uses

    reference: reference flux value for continuum
    """

    # ----------------- Clip the flux array
    '''
    Sigma clip the spectrum (flux) so we don't try to fit noise. 
    c_flux is the flux array with the clipped flux values removed. 
    min_t is the lower flux threshold, and max_t is the upper flux threshold that spst.sigmaclip used in clipping.
    '''
    c_flux, min_t, max_t = spst.sigmaclip(flux, low=clo, high=chi)

    # ----------------- Extract the spectrum data that is within the clipping threshold values
    good = np.where((flux >= min_t) & (flux <= max_t))[0]
    c_wave = wave[good]

    # ----------------- Check that the window is okay, if not, shorten it to an okay length
    if len(c_wave) <= window:
        close = len(c_wave - 10)
        window = int(np.ceil(close) // 2 * 2 + 1)

    # ----------------- Rather than fitting a polynomial, heavily smooth the spectrum and fit to the smoothed curve
    filt = spsi.savgol_filter(c_flux, window, step)

    # ----------------- Interpolate the filtered data back onto the original wavelength grid
    f = interp1d(c_wave, filt, bounds_error=False, fill_value='extrapolate')
    curve = f(wave)

    # -----------------
    '''
    Shift the polynomial by the standard deviation of the flux array. 
    The goal here is to set the continuum closer to 1.0 (rather than the mean of the spectrum)
    '''
    cont = curve + np.std(c_flux)

    median = np.median(cont)
    scale = reference / median
    cont = scale * cont

    return c_wave, c_flux, curve, cont


# -----------------------------------------------------------------------------------------------------------------------
def spec_norm(spectra, spath,
              order_info,
              clips, window, step, reference,
              plot_orig=False,
              plot_clip=False,
              plot_norm=False,
              save_norm=False,
              out_path=None):
    """
    Original Spectrum Continuum Normalization (written by C. Kielty)
    -------------------
    Normalize spectra with a sliding box asymmetric clipping routine

    Best For: Standardize OPERA

    Inputs:
    --------

    spectra: A list of spectra to normalize
    spath: The path to the spectra to normalize
    order_info: A file containing some information where order overlaps roughly occur
    clips: [lower, upper] the lower and upper sigma to clip
    window: Size of the sliding boxcar in pixels; needs to be an odd-number
    step: some value that the Savitzky-Golay filter uses
    reference: reference flux value for continuum

    """

    # ----------------- Read in information about where the orders overlap
    '''
    Earlier a file containing information about order overlaps was generated.  
    '''
    orders = np.load(order_info, allow_pickle=True)
    olaps = []
    for o in orders:
        stop = [o[-1]]
        for i in range(len(o) - 2):
            stop.append([o[i][1], o[i + 1][1]])
        olaps.append(stop)

    # For each order overlap find the mean start and stop wavelength of that order
    orders = []
    for i in range(1, len(olaps[0])):
        sts = []
        sps = []
        for o in olaps:
            sts.append(o[i][0])
            sps.append(o[i][1])
        start = np.mean(sts)
        stop = np.mean(sps)
        pair = [start, stop]
        orders.append(pair)

    # ----------------- For each spectrum
    for spectrum in spectra:

        # ----------------- Read in the spectrum
        data = read_spec(os.path.join(spath, spectrum), ftype='bin')
        wave = data[0]
        flux = data[1]
        err = data[2]

        # ----------------- Plot Original Spectrum
        if plot_orig:
            fig1 = plt.figure(figsize=(8, 5))
            ax1 = fig1.add_subplot(111)
            ax1.plot(wave, flux, color='k', linewidth=0.5)
            ax1.set_title('plot_orig=True\nOriginal Spectrum and Orders')
            plt.xlabel('Wavelength [$\AA$]')
            plt.ylabel('Flux')
            ax1.set_ylim(-5, 5)

        # ----------------- Plot Clipped Spectrum
        if plot_clip:
            fig2 = plt.figure(figsize=(8, 5))
            ax2 = fig2.add_subplot(111)
            plt.xlabel('Wavelength [$\AA$]')
            plt.ylabel('Flux')

        # ----------------- Using the mean order overlap start and stop, find the start and stop of each order
        good_order = find_order_splits(wave, flux, orders)

        # ----------------- Output arrays for the normalized spectrum
        flat_flux = np.array([])
        flat_wave = np.array([])
        flat_err = np.array([])

        # -----------------  For each order in the spectrum...
        for i in range(1, len(good_order)):

            # ----------------- Plot Original
            if plot_orig:
                # If plotting, highlight where the order overlaps occur
                ax1.axvline(good_order[i][0], ls=':', c='g')
                ax1.axvline(good_order[i][1], ls=':', c='r')

            # -----------------  Isolate each order to perform a localized continuum normalization
            short_wave, short_flux, short_err = order_chop(wave, flux, err, good_order[i][0], good_order[i][1])

            # ----------------- Sigma-Clip
            '''
            Using a sigma-clipping routine, clip the short spectrum and use a Savitzky-Golay filter to determine 
            the global structure of the short spectrum. 
            This seems to work better than a polynomial as it handles "jumps" better
            '''
            c_wave, c_flux, curve, cont = sig_clip(short_wave, short_flux, clips[0], clips[1], window, step, reference)

            # ----------------- Plot Clipped Spectrum
            if plot_clip:
                ax2.plot(short_wave, short_flux, color='k', linewidth=0.5, label='Order')
                ax2.plot(c_wave, c_flux, color='tab:green', linewidth=0.5, label='Clipped')
                ax2.plot(short_wave, curve, color='tab:blue', linewidth=0.5, label='Curve')
                ax2.plot(short_wave, cont, color='tab:red', linewidth=0.5, label='Continuum')

            # ----------------- Normalize the order
            # Divide the order by the continuum to normalize that order
            flat = np.array(short_flux / cont)

            # ----------------- Concatenate the order onto the output arrays
            flat_flux = np.concatenate([flat_flux, flat])
            flat_wave = np.concatenate([flat_wave, short_wave])
            flat_err = np.concatenate([flat_err, short_err])

        # ----------------- Plot Clipped Spectrum
        if plot_clip:
            ax2.set_title('plot_clip=True\nOrder by Order Curves: %s' % spectrum)
            ax2.set_ylim(-5, 5)

        # ----------------- Plot Normalized Spectrum
        if plot_norm:
            plt.figure(figsize=(8, 5))
            plt.plot(flat_wave, flat_flux, color='k', linewidth=0.5)
            plt.title('plot_norm=True\nFinal normalized: %s' % spectrum)
            plt.xlabel('Wavelength [$\AA$]')
            plt.ylabel('Flux')
            plt.axhline(1.0, ls=':', c='gray')
            plt.ylim(-5, 5)

        # ----------------- Save Continuum Normalized Spectrum
        if save_norm:
            name = os.path.splitext(spectrum)[0]
            data = [flat_wave, flat_flux, flat_err]
            colheads = ['wave', 'flux', 'err']
            table = Table(data, names=colheads)
            pickle.dump(table, open(out_path + name + '.norm.bin', 'wb'))


# -----------------------------------------------------------------------------------------------------------------------
def find_continuum(obswave,
                   obsflux,
                   linewaves,
                   ref_wave=None,
                   line_width=0.5,
                   wave_range=5.0,
                   trim_sigma=[2.0, 2.0],
                   retrim=True):
    """
    Find wavelength and flux value of the spectrum which constitute the continuum (written by C. Kielty)
    --------------
    A function to identify the wavelengths and fluxes of the continuum near a spectral line.
    This is the old / original version written by Colin Kielty

    :param obswave: (array) Wavelength array of the spectrum

    :param obsflux: (array) Flux array corresponding to the wavelength array of the spectrum

    :param linewaves: (list/array or float) Wavelength(s) of the spectral lines in question

    :param ref_wave:

    :param line_width: (float) Estimated width of the spectral line in wavelength units (angstroms)

    :param wave_range: (float) In the case of a single line, width of window around the line. In the case of several
    lines, width from the lines with the minimum and maximum wavelengths.  In units of wavelength (angstroms)

    :param trim_sigma: [Upper bound, lower bound].  For use in attempting to remove additional lines when identifying
    the continuum (retrim = True).  Keep only what is within [Upper bound, lower bound] sigma from the mean as the
    continuum.  An appropriate choice of trim_sigma will depend on the signal-to-noise of the spectrum

    :param retrim: (True/False) Attempt to trim additional lines from the continuum ?
    :return:
    """

    # ----------------  Check if we are fitting one line or many
    # True if many lines (list, tuple, np.ndarray), False if single line (float)
    are_lines = isinstance(linewaves, (list, tuple, np.ndarray))

    # ---------------- Define a window around the line(s). Large enough to have continuum, small enough to minimize
    # effects of other lines
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
        cont_wave = cont_wave[good]

    # ---------------- Return the continuum wavelengths and fluxes
    return cont_wave, cont_flux


# -----------------------------------------------------------------------------------------------------------------------
def find_continuum2(wave, flux, sigma_lower, sigma_upper):
    """
    Find wavelength and flux value of the spectrum which constitute the continuum (written by J. Glover)
    --------------
    A function to identify the wavelengths and fluxes of the continuum.
    This is a newer version of the code which uses asymmetric sigma clipping

    Best For:  Spectra which have already been continuum normalized such that they are flat.

    :param wave: (array) An array of wavelength values
    :param flux: (array) An array of corresponding flux values
    :param sigma_lower: (float) The lower sigma value
    :param sigma_upper: (float) The upper sigma value

    :return: clipped wavelength and flux arrays
    (Points with flux outside the specified sigma limits have been removed)
    """

    sol = sigma_clip(flux, sigma_lower=sigma_lower, sigma_upper=sigma_upper, maxiters=None)

    clip_flux = [flux[i] for i in range(len(sol.mask)) if sol.mask[i] == False]
    clip_wave = [wave[i] for i in range(len(sol.mask)) if sol.mask[i] == False]

    # ---------------- Return the continuum wavelengths and fluxes
    return clip_wave, clip_flux

