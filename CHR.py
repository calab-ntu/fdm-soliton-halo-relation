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


class CHR_calculator():
    
    def __init__(self,cosmology_model):
        self.cosmo                = cosmology.setCosmology(cosmology_model)
        self.H0                   = cosmology.getCurrent().H0*1e-3                         # km/s/kpc
        self.h                    = self.H0*10                                             # 100km/s/Mpc
        self.omega_M0             = cosmology.getCurrent().Om0
        self.background_density_0 = self.omega_M0*3*(self.H0/kpc2km)**2/(8*np.pi*newton_G) # Msun/kpc^3
    
    def theo_TH_Mc(self,current_redshift, Mh, particle_mass):
        """
        Calculates the theoretical core mass for a soliton halo in FDM using top-hat collapse.
        Assume the density and velocity are evenly distributed in halo.
        Schive2014b https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.261302

        Args:
            redshift (float)      : Redshift.
            halo_mass (float)     : Halo mass in Msun.
            particle_mass (float) : Particle mass in eV.

        Returns:
            mc (float)            : Theoretical core mass for the soliton halo.
        """
        
        current_time_a = redshift_to_a(current_redshift)
        zeta           = get_zeta(current_redshift, self.omega_M0)
        
        Rh             = (3*Mh/(4*np.pi*zeta*(self.background_density_0/current_time_a**3)))**(1/3)
        Ep             = -3/5*newton_G*Mh**2/Rh
        Ek             = -Ep/2
        v              = (2*Ek/Mh)**0.5
        mc             = v*soliton_m_div_v(particle_mass)

        return mc

    def revised_theo_c_FDM_Mc(self,current_redshift, Mh, particle_mass):
        """
        Calculates the revised theoretical core mass for a soliton halo in FDM.

        Args:
            redshift (float)      : Redshift.
            halo_mass (float)     : Halo mass in Msun.
            particle_mass (float) : Particle mass in eV.

        Returns:
            mc (float)            : Revised theoretical core mass for the soliton halo.
        """
        
        current_time_a = redshift_to_a(current_redshift)
        c_theo         = concentration_para_FDM(Mh, current_redshift, self.h, particle_mass)
        zeta           = get_zeta(current_redshift, self.omega_M0)
        Rh             = (3*Mh/(4*np.pi*zeta*(self.background_density_0/current_time_a**3)))**(1/3)

        def f_c(c):
            return (2*c*(1+c)*np.log(1+c)-2*c**2-c**3)/((1+c)*np.log(1+c)-c)**2

        Ep             = newton_G*Mh**2/Rh/2*f_c(c_theo)
        Ek             = -Ep/2
        halo_average_v = (2*Ek/Mh)**0.5
        inner_halo_v   = halo_average_v*temp_from_c(current_redshift, Mh, self.h, particle_mass)
        soliton_v      = inner_halo_v*s_h_eq

        mc             = soliton_v*soliton_m_div_v(particle_mass)
        
        return mc


def FDM_supress_laroche(M, particle_mass):
    """
    Calculates the LaRoche suppression factor for FDM using corrected values from Kawai (2024).
    Kawai2024 identified typos in the a, b, c values of LaRoche (2022) and refitted them.
    Laroche2022 https://academic.oup.com/mnras/article/517/2/1867/6711699
    Kawai2024 https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.023519

    Args:
        halo_mass (float)     : Halo mass in Msun.
        particle_mass (float) : Particle mass in eV.

    Returns:
        F (float)             : LaRoche suppression factor.
    """

    def half_mode_mass(particle_mass):
        """
        Returun half mode mass. (Mass below which small structures are suppressed due to FDM wave-like behavior)
        Schive2016 eq.6 https://iopscience.iop.org/article/10.3847/0004-637X/818/1/89

        Args:
            particle_mass (float) : Particle mass in eV.

        Returns:
            h_m_mass (float)      : Half mode mass in Msun.
        """

        h_m_mass = 3.8e10*(particle_mass/1e-22)**(-4/3)

        return h_m_mass

    a , b, c = 5.496, -1.648, -0.417
    x        = M/half_mode_mass(particle_mass)
    F        = (1+a*x**b)**c

    return F

def concentration_para_CDM(halo_mass, redshift):
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
    
    c_CDM = concentration.concentration(halo_mass, 'vir', redshift, model = 'ishiyama21',halo_sample ='relaxed')
    
    return c_CDM

def concentration_para_FDM(halo_mass, redshift, h, particle_mass):
    """
    Calculates the halo concentration for FDM with Laroche suppression.

    Args:
        halo_mass (float)     : Halo mass in Msun.
        redshift (float)      : Redshift.
        particle_mass (float) : Particle mass in eV.

    Returns:
        c_FDM (float)         : Prediction of halo concentration for FDM.
    """
    
    c_CDM     = concentration_para_CDM(halo_mass*h, redshift)
    F_supress = FDM_supress_laroche(halo_mass, particle_mass)
    c_FDM     = c_CDM*F_supress
    
    return c_FDM

def soliton_dens(x, core_radius, particle_mass):
    """
    Calculates the soliton density profile in physical frame.
    Schive2014a Supplement eq.4 https://arxiv.org/abs/1406.6586

    Args:
        x (float)             : radius in kpc
        core_radius (float)   : Core radius in kpc.
        particle_mass (float) : Particle mass in eV.

    Returns:
        density (float)       : Soliton density at the given radius in Msun/kpc**3.
    """
    
    density = ((1.9*(particle_mass/10**-23)**-2*(core_radius**-4))/((1 + 9.1*10**-2*(x/core_radius)**2)**8))*10**9
    
    return density

def grad_soliton(x, core_radius, particle_mass):
    """
    Calculates the gradient of soliton core density profile in physical frame.

    Args:
        x (float)             : Radius in kpc.
        core_radius (float)   : Core radius in kpc.
        particle_mass (float) : Particle mass in eV.

    Returns:
        dens_gradient (float) : The gradient of soliton density at the given radius in Msun/kpc^4.
    """
    
    dens_gradient = (1.9*(particle_mass/10**-23)**-2*(core_radius**-4)*(-9.1*10**-2*16*x/core_radius**2)/((1 + 9.1*10**-2*(x/core_radius)**2)**9))*10**9
    
    return dens_gradient


def soliton_m_div_v(particle_mass, enclose_r = 3.3):
    """
    Calculates the soliton core mass divided by its enclosed average velocity in physical frame.
    This value is proportional to a given particle mass.

    Args:
        particle_mass (float)  : Particle mass in eV.
        enclose_r (float)      : The enclosed radius, with a default value of 3.3 times the core radius. This value typically encloses 95% of the energy.

    Returns:
        m_divided_by_v (float) : Soliton's mass / velcoity in Msun/kpc/s
    """
    
    core_radius = 4 # kpc. Any core_radius value can be used to evaluate the constant.
    
    def shell_mass(r, particle_mass):
        """
        Calculates the shell mass at radius r.

        Args:
            r (float)             : radius in kpc.
            particle_mass (float) : Particle mass in eV.

        Returns:
            mass (float)          : Shell mass of the soliton at radius r in  Msun/kpc.
        """
        
        mass = 4*np.pi*r**2*soliton_dens(r, core_radius, particle_mass)

        return mass
    
    def Ek_func(r, particle_mass):
        """
        Calculates the kinetic energy Ek in physical frame.

        Args:
            r (float)             : radius in kpc.
            particle_mass (float) : Particle mass in eV.

        Returns:
            Ek (float)            : Kinetic energy in this shell in Msun*kpc/s^2
        """
        
        v = 0.5/soliton_dens(r, core_radius, particle_mass)*(hbar/(particle_mass*eV_c2))*grad_soliton(r, core_radius, particle_mass)
        Ek = 0.5*shell_mass(r, particle_mass)*v**2
        
        return Ek
    
    ms             = quad(lambda r: shell_mass(r, particle_mass), 0, core_radius*enclose_r)[0] # Msun
    Eks            = quad(lambda r: Ek_func(r, particle_mass), 0, core_radius*enclose_r)[0]    # Msun*kpc^2/s^2

    vs             = (2*Eks/ms)**0.5                                                     # kpc/s
    mc             = quad(lambda r: shell_mass(r, particle_mass), 0, core_radius)[0]     # Msun
    m_divided_by_v = mc/vs

    return m_divided_by_v


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


def temp_from_c(current_redshift, Mh, h, particle_mass):
    """
    Calculates the temperature ratio (velocity of inner halo / velocity of whole halo average) from the halo concentration for FDM by empirical fitting.

    Args:
        redshift (float)      : Redshift.
        halo_mass (float)     : Halo mass in Msun.
        particle_mass (float) : Particle mass in eV.

    Returns:
        temp_ratio (float)    : The temperature ratio between the average halo and the center halo.
    """
    
    c_FDM      = concentration_para_FDM(Mh, current_redshift, h, particle_mass)
    temp_ratio = 0.27*np.log10(c_FDM)+1
    
    return temp_ratio


if __name__ == '__main__':


    ### load the command-line parameters to input your halo mass, particle mass, and redshift

    parser = argparse.ArgumentParser( description='Predicting the core-halo mass relation in fuzzy dark matter (FDM)' )

    parser.add_argument( '--halo_mass',     action='store', required=False, type=float, dest='halo_mass',
                        help='halo mass (Msun)',      default=1e12 )
    parser.add_argument( '--redshift',      action='store', required=False, type=float, dest='redshift',
                        help='redshift',              default=0 )
    parser.add_argument( '--particle_mass', action='store', required=False, type=float, dest='particle_mass',
                        help='paticle mass (eV)', default=2e-23 )

    args=parser.parse_args()

    halo_mass           = args.halo_mass
    current_redshift    = args.redshift
    particle_mass       = args.particle_mass
    
    ### set cosmology
    # Initialize a CHR_calculator class. You can change to other cosmology
    chr_calculator = CHR_calculator('planck18')
    print(chr_calculator.cosmo.name)

    print(f"halo mass: {halo_mass:.2e}, redshift: {current_redshift:.2e}, particle mass: {particle_mass:.2e}")

    ### Calculate the revised core mass
    revised_c_FDM_Mc = chr_calculator.revised_theo_c_FDM_Mc(current_redshift, halo_mass, particle_mass)
    print(f"Predicted core mass (this work)  : {revised_c_FDM_Mc:.2e}")

    ### Calculate the Schive2014 core mass
    theo_Mc = chr_calculator.theo_TH_Mc(current_redshift, halo_mass, particle_mass)
    print(f"Predicted core mass (Schive2014) : {theo_Mc:.2e}")