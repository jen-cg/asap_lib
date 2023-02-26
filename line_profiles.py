import numpy as np
from scipy import integrate

"""
Line profile functions 
"""


def Gauss(x, a, b, c):
    """
    A Gaussian line profile
    :param x: wavelength range
    :param a: amplitude parameter
    :param b: shift parameter
    :param c: width parameter
    :return:
    """
    return 1 - a * np.e ** (- (x - b) ** 2 / (2 * c ** 2))


# ---------------------
def GaussAnalyticArea(a, c):
    """

    :param a: amplitude parameter
    :param c: width parameter
    :return:
    """
    return np.sqrt(2 * np.pi) * np.abs(a) * np.abs(c)


# ---------------------
def GaussArea(xmin, xmax, a, b, c, cont=1):
    """

    :param xmin: beginning of integration range
    :param xmax: end of integration range
    :param a: amplitude parameter
    :param b: shift parameter
    :param c: width parameter
    :param cont:
    :return:
    """
    A1 = cont * (xmax - xmin)
    A2, err = integrate.quad(Gauss, xmin, xmax, args=(a, b, c))
    return A1 - A2, err


# -----------------------------------------------------------------------------------------------------------------------
def Lorentz(x, a, b, c):
    """
    A Lorentz line profile

    :param x: wavelength range
    :param a: amplitude parameter
    :param b: shift parameter
    :param c: width parameter
    :return:
    """
    return 1 - a * ((x - b) ** 2 + c ** 2) ** -1


# ---------------------
def LorentzAnalyticArea(a, c):
    """
    The analytic area under a Lorentz profile line when integrated over all space
    :param a: amplitude parameter
    :param c: width parameter
    :return:
    """
    return a * np.pi / c


# ---------------------
def LorentzArea(xmin, xmax, a, b, c, cont=1):
    """
    The area of a Lorentz profile line when integrated over a certain wavelength range
    (numerical)

    :param xmin: beginning of integration range
    :param xmax: end of integration range
    :param a: amplitude parameter
    :param b: shift parameter
    :param c: width parameter
    :param cont: location of the continuum
    :return: area, error
    """
    A1 = cont * (xmax - xmin)
    A2, err = integrate.quad(Lorentz, xmin, xmax, args=(a, b, c))
    return A1 - A2, err


# -----------------------------------------------------------------------------------------------------------------------
def PseudoVoigt(x, eta, b, al, cl, ag, cg):
    """
    A Pseudo-Voigt line profile

    :param x: wavelength range
    :param eta: weighting between Gaussian and Lorentz parts
    :param b: shift parameter
    :param al: Lorentz amplitude parameter
    :param cl: Lorentz width parameter
    :param ag: Gaussian amplitude parameter
    :param cg: Gaussian width parameter

    :return:
    """
    return eta * Lorentz(x, al, b, cl) + (1 - eta) * Gauss(x, ag, b, cg)


# ---------------------
def PseudoVoigtAnalyticArea(eta, b, al, cl, ag, cg):
    """

    :param eta: weighting between Gaussian and Lorentz parts
    :param b: shift parameter
    :param al: Lorentz amplitude parameter
    :param cl: Lorentz width parameter
    :param ag: Gaussian amplitude parameter
    :param cg: Gaussian width parameter
    :return:
    """
    return eta * LorentzAnalyticArea(al, cl) + (1 - eta) * GaussAnalyticArea(ag, cg)


# ---------------------
def PseudoVoigtArea(xmin, xmax, eta, b, al, cl, ag, cg, cont=1):
    """

    :param xmin: beginning of integration range
    :param xmax: end of integration range
    :param eta:  weighting between Gaussian and Lorentz parts
    :param b:  shift parameter
    :param al:  Lorentz amplitude parameter
    :param cl:  Lorentz width parameter
    :param ag:  Gaussian amplitude parameter
    :param cg:  Gaussian width parameter
    :param cont:
    :return:
    """
    A1 = cont * (xmax - xmin)
    A2, err = integrate.quad(PseudoVoigt, xmin, xmax, args=(eta, b, al, cl, ag, cg))
    return A1 - A2, err
