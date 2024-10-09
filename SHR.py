#!/usr/bin/env python3

import argparse
import sys

import colossus
import numpy as np
from colossus.cosmology import cosmology
from colossus.halo import concentration
from colossus.utils import constants
from scipy.integrate import quad

### constant
kpc2km               = constants.KPC*1e-5                                     # km
hbar                 = constants.H*0.5/np.pi/constants.KPC**2/constants.MSUN  # kpc^2 msun/s
eV_c2                = constants.EV/constants.C**2/constants.MSUN             # Msun
newton_G             = constants.G/kpc2km**2                                  # (kpc^3)/(s^2*Msun)

### the velocity ratio between soliton and inner halo by empirical fitting
s_h_eq = 0.5**0.5*0.8935555051894757


class SHR_calculator():
    
    def __init__(self,cosmology_model):
        self.cosmo                = cosmology.setCosmology(cosmology_model)
        self.H0                   = cosmology.getCurrent().H0*1e-3                         # km/s/kpc
        self.h                    = self.H0*10                                             # 100km/s/Mpc
        self.omega_M0             = cosmology.getCurrent().Om0
        self.background_density_0 = self.omega_M0*3*(self.H0/kpc2km)**2/(8*np.pi*newton_G) # Msun/kpc^3
    
    def theo_TH_Ms(self,current_redshift, Mh, m22):
        """
        Calculates the theoretical soliton mass for a halo in FDM using top-hat collapse.
        Assume the density and velocity are evenly distributed in halo.
        Schive2014b https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.261302

        Args:
            redshift (float)      : Redshift.
            halo_mass (float)     : Halo mass in Msun.
            m22 (float) : Particle mass in 1e-22 eV.

        Returns:
            ms (float)            : Theoretical soliton mass for the soliton halo in Msun.
        """
        
        current_time_a = redshift_to_a(current_redshift)
        zeta           = get_zeta(current_redshift, self.omega_M0)
        zeta_0         = get_zeta(0, self.omega_M0)
        Mmin0 = 4.4e7*m22**(-3/2)

        ms = 0.25*current_time_a**(-0.5)*(zeta/zeta_0)**(1/6)*(Mh/Mmin0)**(1/3)*Mmin0

        return ms

    def revised_theo_c_FDM_Ms(self,current_redshift, Mh, m22):
        """
        Calculates the revised theoretical soliton mass for a halo in FDM.

        Args:
            redshift (float)      : Redshift.
            halo_mass (float)     : Halo mass in Msun.
            m22 (float) : Particle mass in 1e-22 eV.

        Returns:
            ms (float)            : Revised theoretical soliton mass for the halo in Msun.
        """
        
        def f_c(c):
            """
            Coefficient of NFW potential according to concentration parameter.
            Args:
                c (float)         : Concnetration parameter.

            Returns:
                Coefficient.
            """
            return (2*c*(1+c)*np.log(1+c)-2*c**2-c**3)/((1+c)*np.log(1+c)-c)**2

        current_time_a = redshift_to_a(current_redshift)
        c_theo         = concentration_para_FDM(Mh, current_redshift, self.h, m22)
        zeta           = get_zeta(current_redshift, self.omega_M0)
        Rh             = (3*Mh/(4*np.pi*zeta*(self.background_density_0/current_time_a**3)))**(1/3)

        Ep             = newton_G*Mh**2/Rh/2*f_c(c_theo)
        alpha          = 2**(-0.5)
        beta           = temp_from_c(current_redshift, Mh, self.h, m22)
        gamma          = 0.8935555051894757

        ws             = (-Ep/Mh)**0.5*alpha*beta*gamma  # kpc/s
        ms             = 3.15e8*ws*kpc2km/100*m22**-1

        return ms


def FDM_supress_laroche(M, m22):
    """
    Calculates the LaRoche suppression factor for FDM using corrected values from Kawai (2024).
    Kawai2024 identified typos in the a, b, c values of LaRoche (2022) and refitted them.
    Laroche2022 https://academic.oup.com/mnras/article/517/2/1867/6711699
    Kawai2024 https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.023519

    Args:
        halo_mass (float)     : Halo mass in Msun.
        m22 (float) : Particle mass in 1e-22 eV.

    Returns:
        F (float)             : LaRoche suppression factor.
    """

    a , b, c = 5.496, -1.648, -0.417
    x        = M/half_mode_mass(m22)
    F        = (1+a*x**b)**c

    return F

def half_mode_mass(m22):
    """
    Returun half mode mass. (Mass below which small structures are suppressed due to FDM wave-like behavior)
    Schive2016 eq.6 https://iopscience.iop.org/article/10.3847/0004-637X/818/1/89

    Args:
        m22 (float) : Particle mass in 1e-22 eV.

    Returns:
        h_m_mass (float)      : Half mode mass in Msun.
    """

    h_m_mass = 3.8e10*m22**(-4/3)

    return h_m_mass

def concentration_para_CDM(halo_mass, h, redshift):
    """
    Calculates the halo concentration using colossus package.
    Diemer 2015 https://iopscience.iop.org/article/10.1088/0004-637X/799/1/108
    colossus doc https://bdiemer.bitbucket.io/colossus/halo_concentration.html#concentration-models
    
    Args:
        halo_mass (float) : Halo mass in Msun/h.
        redshift (float)  : Redshift.

    Returns:
        c_CDM (float)     : Prediction of halo concentration in CDM model.
    """
    
    c_CDM = concentration.concentration(halo_mass*h, 'vir', redshift, model = 'ishiyama21',halo_sample ='relaxed')
    
    return c_CDM

def concentration_para_FDM(halo_mass, redshift, h, m22):
    """
    Calculates the halo concentration for FDM with Laroche suppression.

    Args:
        halo_mass (float)     : Halo mass in Msun.
        redshift (float)      : Redshift.
        m22 (float) : Particle mass in 1e-22 eV.

    Returns:
        c_FDM (float)         : Prediction of halo concentration for FDM.
    """
    
    c_CDM     = concentration_para_CDM(halo_mass, h, redshift)
    F_supress = FDM_supress_laroche(halo_mass, m22)
    c_FDM     = c_CDM*F_supress
    
    return c_FDM

def get_zeta(redshift, omega_M0):
    """
    Calculates the zeta parameter for a spherical cluster model. 
    Bryan1998 eq.6 https://iopscience.iop.org/article/10.1086/305262
    The Omega_R is assumed to be 0.
    Schive2014b https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.261302

    Args:
        redshift (float) : Redshift.
        omega_M0 (float) : The mass density at the redshift = 0.

    Returns:
        z (float)        : zeta parameter.
    """

    omega_M = (omega_M0*(1 + redshift)**3)/(omega_M0*(1 + redshift)**3 + (1 - omega_M0))
    zeta    = (18*np.pi**2 + 82*(omega_M - 1) - 39*(omega_M - 1)**2)/omega_M 
    
    return zeta


def redshift_to_a(redshift):
    """
    Calculates scale factor a from redshift.

    Args:
        redshift (float) : Redshift.

    Returns:
        a (float)        : Scale factor a.
    """
    
    a = 1/(1+redshift)
    
    return a


def temp_from_c(current_redshift, Mh, h, m22):
    """
    Calculates the temperature ratio (velocity of inner halo / velocity of whole halo average) from the halo concentration for FDM by empirical fitting.

    Args:
        redshift (float)      : Redshift.
        halo_mass (float)     : Halo mass in Msun.
        m22 (float) : Particle mass in 1e-22 eV.

    Returns:
        temp_ratio (float)    : The temperature ratio between the average halo and the center halo.
    """
    
    c_FDM      = concentration_para_FDM(Mh, current_redshift, h, m22)
    temp_ratio = 0.27*np.log10(c_FDM)+1.05
    
    return temp_ratio


if __name__ == '__main__':


    ### load the command-line parameters to input your halo mass, particle mass, and redshift

    parser = argparse.ArgumentParser( description='Predicting the soliton-halo mass relation in fuzzy dark matter (FDM)' )

    parser.add_argument( '-hm',  '--halo_mass',     action='store', required=False, type=float, dest='halo_mass',
                        help='halo mass (Msun)',      default=1e12 )
    parser.add_argument( '-z',   '--redshift',      action='store', required=False, type=float, dest='redshift',
                        help='redshift',              default=0 )
    parser.add_argument( '-m22', '--m22',           action='store', required=False, type=float, dest='m22',
                        help='paticle mass (1e-22 eV)', default=2e-1 )

    args=parser.parse_args()

    halo_mass           = args.halo_mass
    current_redshift    = args.redshift
    m22                 = args.m22
    
    ### set cosmology
    # Initialize a SHR_calculator class. You can change to other cosmology
    shr_calculator = SHR_calculator('planck18')
    print(shr_calculator.cosmo.name)

    print(f"halo mass: {halo_mass:.2e}, redshift: {current_redshift:.2e}, m22: {m22:.2e}")

    ### Calculate the revised soliton mass
    revised_c_FDM_Ms = shr_calculator.revised_theo_c_FDM_Ms(current_redshift, halo_mass, m22)
    print(f"Predicted soliton mass (this work)  : {revised_c_FDM_Ms:.2e}")

    ### Calculate the Schive2014 soliton mass
    theo_Ms = shr_calculator.theo_TH_Ms(current_redshift, halo_mass, m22)
    print(f"Predicted soliton mass (Schive2014) : {theo_Ms:.2e}")