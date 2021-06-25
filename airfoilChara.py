""" Returns alpha0 (in deg) and CMc/4 for different types of airfoils """

import numpy as np
import scipy.special as sc


def poly(coeffs, flip=False):
    """ Polynomial airfoil """
    """ In the order [a0, a1, a2, ... ] for a0 + a1*x + a2*x^2 + ..."""
    if flip == True:
        coeffs = np.flip(coeffs)
    alpha0 = 0.0
    CMcby4 = 0.0
    for i, ai in enumerate(coeffs[1:], start=1):
        alpha0 += i*ai*sc.beta(i+0.5, 0.5)
        CMcby4 += i*ai*(sc.beta(i+1.5, 0.5)-3.0*sc.beta(i+0.5, 1.5))

    return alpha0*2.0*180.0/(np.pi)**2.0, CMcby4

def bern(coeffs):
    """ Bernstein polynomial airfoil """

    alpha0 = 0.0
    CMcby4 = 0.0
    n = len(coeffs)-1
    for i, ai in enumerate(coeffs, start=0):
        k = n-i
        infAlpha0 = sc.comb(n, i)*(i*sc.beta(i+1/2, k+1/2) - \
                          k*sc.beta(i+3/2, k-1/2))
        alpha0 += ai*infAlpha0
        infCM = sc.comb(n, i)*(1*i*sc.beta(i+1/2, k+1/2) - \
                            4*i*sc.beta(i+1/2, k+3/2) - \
                            1*k*sc.beta(i+3/2, k-1/2) + \
                            4*k*sc.beta(i+3/2, k+1/2))
        CMcby4 += ai*infCM

    alpha0 = alpha0*2/np.pi*(180.0/np.pi)
    return alpha0, CMcby4

def naca4(Mdigit, Pdigit):
    """ NACA 4-digit airfoil """

    M = Mdigit/100.0
    P = Pdigit/10.0

    if M == 0:
        alpha0 = 0.0
        CMcby4 = 0.0
    else:
        thetaP = np.arccos(1.0-2.0*P)
        alpha0 = np.pi*P**2*(3-4*P)+(1-2*P)*(thetaP*(3-4*P)+(2*P-3)*np.sin(thetaP))
        alpha0 *= -M/(np.pi*2*P*P*(1.0-P)**2.0)
        CMcby4 = M/(P*P*(1-P)**2)* \
                (np.sin(thetaP)/12*(16*P**3-12*P*P-4*P+3)+((2*P-1)*thetaP-np.pi*P*P)/4)

    return alpha0*180.0/np.pi, CMcby4

def naca5(Ldigit, Pdigit, Qdigit):
    """ NACA 5-digit airfoil """

    if Ldigit != 2:
        return ValueError('Only L=2 implemented')

    Pdigit = Pdigit/20
    isstandard = False
    if Qdigit == 0:
        isstandard = True
    else:
        isstandard = False

    if isstandard:
        r = 3.33333333333212*(Pdigit**3) + \
                0.700000000000909*(Pdigit**2) + \
                1.19666666666638*Pdigit - \
                0.00399999999996247;
        k1 = 1514933.33335235*(Pdigit**4) - \
                1087744.00001147*(Pdigit**3) + \
                286455.266669048*(Pdigit**2) - \
                32968.4700001967*Pdigit + \
                1420.18500000524;
        thetar = np.arccos(1-2*r)
        alpha0 = -k1/(48*np.pi)*(8*np.pi*r**3 + \
                                thetar*(-24*r**2+36*r-15) + \
                                np.sin(thetar)*(8*r**2-26*r+15))
        CMcby4 = -k1/192*(3*thetar*(8*r-5) + \
                                  np.sin(thetar)*(16*r**3 - \
                                                  8*r**2-14*r+15))
    else:  # reflexed
        r = 10.6666666666861*(Pdigit**3) - \
                2.00000000001601*(Pdigit**2) + \
                1.73333333333684*Pdigit - \
                0.0340000000002413
        k1 = -27973.3333333385*(Pdigit**3) + \
                17972.8000000027*(Pdigit**2) - \
                3888.40666666711*Pdigit + \
                289.076000000022
        k2_k1 = 85.5279999999984*(Pdigit**3) - \
                34.9828000000004*(Pdigit**2) + \
                4.80324000000028*Pdigit - \
                0.21526000000003
        thetar = np.arccos(1-2*r)
        alpha0 = k1/(48*np.pi)*((1-k2_k1)*(thetar*(24*r**2-36*r+15) \
                               - np.sin(thetar)*(8*r**2-26*r+15)) \
                               +(np.pi*k2_k1) \
                                *(8*r**3-12*r+7)-8*np.pi*r**3)
        CMcby4 = -k1/192*((1-k2_k1)*(3*thetar \
                                     *(8*r-5)+np.sin(thetar) \
                                     *(16*r**3-8*r**2-14*r+15)) \
                          + 3*np.pi*k2_k1*(8*r-5))

    alpha0 *= 180/np.pi
    return alpha0, CMcby4
