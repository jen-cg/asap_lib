import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate

"""
Equivalent Width Functions 

- Functions for calculating EW
"""


# -----------------------------------------------------------------------------------------------------------------------
def calculateEW_mc(w, f, a, b, delta_w, delta_f, N=100):
    """
    Calculate the EW of a spectral line using N iterations of a Monte-Carlo routine

    - w wavelength of array
    - f flux array

    - a wavelength of the left end of integration
    - b wavelength of the right end of integration

    - delta_w width of wavelength box
    - delta_f width of flux box

    - N number of iterations of the routine
    """

    # ------------------ For N iterations
    A_sample = []
    for i in range(0, N):
        # -------- Randomly sample a point in the box
        L_w = np.random.uniform(a - delta_w / 2, a + delta_w / 2)
        L_f = np.random.uniform(1 - delta_f / 2, 1 + delta_f / 2)

        R_w = np.random.uniform(b - delta_w / 2, b + delta_w / 2)
        R_f = np.random.uniform(1 - delta_f / 2, 1 + delta_f / 2)

        # -------- Connect points with a straight line
        slope = (L_f - R_f) / (L_w - R_w)
        intercept = L_f - slope * L_w

        x = np.linspace(L_w, R_w, 1000)
        y = slope * x + intercept

        # -------- Integrate under the line
        A1 = integrate.cumtrapz(y, x)[-1]

        # -------- Interpolate the spectrum onto the same grid as the line
        func = interp1d(w, f)

        # -------- Intergrate under the spectrum (from sample end point to sample end point)
        A2 = integrate.cumtrapz(func(x), x)[-1]

        # -------- Find the area of the spectral line
        A_sample.append(A1 - A2)

    return A_sample


# -----------------------------------------------------------------------------------------------------------------------
def systematics_MC(w, f, start_w, delta_w, delta_f, diff_max=2, N=100):
    """
    Calculate the EW of a spectral line using N iterations of a Monte-Carlo routine
    but vary the wavelength to start the integration at

    - a wavelength of the left end of integration
    - b wavelength of the right end of integration

    - N number of iterations of the routine

    :param w: wavelength of array
    :param f: flux array
    :param start_w: wavelength to start the integration at.
    Should be ~ wavelength of the spectral feature
    :param delta_w: delta_w width of wavelength box
    :param delta_f: delta_f width of flux box
    :param diff_max: The maximum distance from the start wavelength to integrate to
    :param N: number of iterations of the MC routine
    :return: array of EW, corresponding array of integration end points
    """
    diffs = np.linspace(0, diff_max)

    A = []
    for diff in diffs:
        # ------------------ Pick ends of line profile
        a = start_w - diff
        b = start_w + diff

        # ------------------
        A_sample = calculateEW_mc(w, f, a, b, delta_w, delta_f, N=N)

        A.append(np.mean(A_sample))

    return A, diffs
