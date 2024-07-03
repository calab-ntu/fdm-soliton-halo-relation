import argparse
import sys

import colossus
import numpy as np
from colossus.cosmology import cosmology
from colossus.halo import concentration
from colossus.utils import constants
from scipy.integrate import quad

### constant
kpc2km               = constants.KPC/1e5 # km
hbar                 = constants.H/2/np.pi/constants.KPC**2/constants.MSUN  # kpc^2 msun/s
eV_c2                = constants.EV/constants.C**2/constants.MSUN # Msun
newton_G             = constants.G/kpc2km**2 # (kpc^3)/(s^2*Msun)

cosmo = cosmology.setCosmology('planck18')
H0                   = cosmology.getCurrent().H0/1000 # km/s/kpc
h                    = H0*10 # 100km/s/Mpc
omega_M0             = cosmology.getCurrent().Om0
background_density_0 = omega_M0*3*(H0/kpc2km)**2/(8*np.pi*newton_G) # Msun/kpc^3

def set_cosmology(cosmology_name):
    """
    Sets the current cosmology.
    """
    global cosmo, H0, h, omega_M0, background_density_0
    cosmo = cosmology.setCosmology(cosmology_name)
    H0                   = cosmology.getCurrent().H0/1000 # km/s/kpc
    h                    = H0*10 # 100km/s/Mpc
    omega_M0             = cosmology.getCurrent().Om0
    background_density_0 = omega_M0*3*(H0/kpc2km)**2/(8*np.pi*newton_G) # Msun/kpc^3

    # return cosmo, H0, h, omega_M0, background_density_0


def FDM_supress_laroche(M, particle_mass):
    """
    Calculates the LaRoche suppression factor for FDM.
    https://academic.oup.com/mnras/article/517/2/1867/6711699

    Args:
        halo_mass (float): Halo mass in solar masses.
        particle_mass (float): Particle mass in solar masses.

    Returns:
        float: LaRoche suppression factor.
    """

    def half_mode_mass(particle_mass):
        ### https://iopscience.iop.org/article/10.3847/0004-637X/818/1/89
        return 3.8e10*(particle_mass/1e-22)**(-4/3)

    a = 5.496
    b = -1.648
    c = -0.417
    x = M/half_mode_mass(particle_mass)
    F = (1+a*x**b)**c
    return F

def concentration_para_CDM(halo_mass, redshift):
    """
    Calculates the halo concentration using colossus package.
    https://iopscience.iop.org/article/10.1088/0004-637X/799/1/108
    Args:
        halo_mass (float): Halo mass in solar masses.
        redshift (float): Redshift.

    Returns:
        float: Halo concentration.
    """
    c_CDM = concentration.concentration(halo_mass*h, 'vir', redshift, model = 'ishiyama21',halo_sample ='relaxed')
    return c_CDM

def concentration_para_FDM(halo_mass, redshift, particle_mass):
    """
    Calculates the halo concentration for FDM with Laroche suppression.

    Args:
        halo_mass (float): Halo mass in solar masses.
        redshift (float): Redshift.
        particle_mass (float): Particle mass in solar masses.

    Returns:
        float: Halo concentration for FDM.
    """
    c_CDM = concentration_para_CDM(halo_mass, redshift)
    F_supress = FDM_supress_laroche(halo_mass, particle_mass)
    return c_CDM*F_supress

def soliton_dens(x, core_radius, particle_mass):
    """
    Calculates the soliton density profile.
    https://www.nature.com/articles/nphys2996
    Args:
        x (float): radius in kpc/
        core_radius (float): core_radius in kpc.
        particle_mass (float): Particle mass in solar masses.

    Returns:
        float: Soliton density at the given radius.
    """
    return ((1.9*(particle_mass/10**-23)**-2*(float(core_radius)**-4))/((1 + 9.1*10**-2*(x/float(core_radius))**2)**8))*10**9

def grad_soliton(x, core_radius, particle_mass):
    """
    Calculates the gradient of soliton core density profile.

    Args:
        x (float): Radius in kiloparsecs.
        core_radius (float): Core radius in kiloparsecs.
        particle_mass (float): Particle mass in solar masses.

    Returns:
        float: The gradient of soliton density at the given radius.
    """
    return (1.9*(particle_mass/10**-23)**-2*(float(core_radius)**-4)*(-9.1*10**-2*16*x/float(core_radius)**2)/((1 + 9.1*10**-2*(x/float(core_radius))**2)**9))*10**9 # Msun/kpc^4


def soliton_m_dev_v(particle_mass): # soliton mass/velocity
    """
    Calculates the soliton enclosed mass devide by its enclosed average velocity.
    This value is a constant with given particle mass.

    Args:
        particle_mass (float): Particle mass in solar masses.

    Returns:
        float: Soliton's mass / velcoity in Msun/kpc/s
    """
    core_radius = 4 #kpc, can be any
    def shell_mass(r, particle_mass):
        return 4*np.pi*r**2*soliton_dens(r, core_radius, particle_mass)
    
    ms = quad(lambda r: shell_mass(r, particle_mass), 0, core_radius*3.3)[0] #Msun

    def Ek_func(r, particle_mass):
        v = 0.5/soliton_dens(r, core_radius, particle_mass)*(hbar/(particle_mass*eV_c2))*grad_soliton(r, core_radius, particle_mass)  #kpc/s
        return 0.5*shell_mass(r, particle_mass)*v**2  # Msun*kpc^2/s^2
    
    Eks = quad(lambda r: Ek_func(r, particle_mass), 0, core_radius*3.3)[0]

    vs = (2*Eks/ms)**0.5
    mc = quad(lambda r: shell_mass(r, particle_mass), 0, core_radius)[0]

    return mc/vs #Msun/kpc/s

def Xi(time_z):
    """
    Calculates the xi parameter for a spherical cluster model.
    https://iopscience.iop.org/article/10.1086/305262

    Args:
        redshift (float): Redshift.

    Returns:
        float: Xi parameter.
    """

    omega_M = (omega_M0*(1 + time_z)**3)/(omega_M0*(1 + time_z)**3 + (1 - omega_M0))
    z = (18*np.pi**2 + 82*(omega_M - 1) - 39*(omega_M - 1)**2)/omega_M 
    return z

def theo_TH_Mc(current_redshift, Mh, particle_mass):
    """
    Calculates the theoretical core mass for a soliton halo in FDM using top-hat collapse.

    Args:
        redshift (float): Redshift.
        halo_mass (float): Halo mass in solar masses.
        particle_mass (float): Particle mass in solar masses.

    Returns:
        float: Theoretical core mass for the soliton halo.
    """
    
    current_time_a = 1/(1+current_redshift)
    zeta = Xi(current_redshift)
    
    Rh = (3*Mh/(4*np.pi*zeta*(background_density_0/current_time_a**3)))**(1/3)
    Ep = -3/5*newton_G*Mh**2/Rh
    Ek = -Ep/2
    v = (2*Ek/Mh)**0.5
    mc = v*soliton_m_dev_v(particle_mass)

    return mc

def soliton_halo_equilibrium():
    """
    return the value of the velocity ratio between soliton and inner halo by emprotical fitting.
    """
    return 0.5**0.5*0.8935555051894757

def temp_from_c(current_redshift, Mh, particle_mass):
    """
    Calculates the temperature from the halo concentration for FDM.

    Args:
        redshift (float): Redshift.
        halo_mass (float): Halo mass in solar masses.
        particle_mass (float): Particle mass in solar masses.

    Returns:
        float: Temperature.
    """
    c_theo = concentration_para_FDM(Mh,current_redshift, particle_mass)
    return 0.27*np.log10(c_theo)+1

def revised_theo_c_FDM_Mc(current_redshift, Mh, particle_mass):
    """
    Calculates the revised theoretical core mass for a soliton halo in FDM.

    Args:
        redshift (float): Redshift.
        halo_mass (float): Halo mass in solar masses.
        particle_mass (float): Particle mass in solar masses.

    Returns:
        float: Revised theoretical core mass for the soliton halo.
    """
    current_time_a = 1/(1+current_redshift)
    c_theo = concentration_para_FDM(Mh,current_redshift,particle_mass)
    zeta = Xi(current_redshift)
    Rs = (3*Mh/(4*np.pi*zeta*(background_density_0/current_time_a**3)))**(1/3)/c_theo

    def f_c(c):
        return (2*c*(1+c)*np.log(1+c)-2*c**2-c**3)/((1+c)*np.log(1+c)-c)**2

    Rh = Rs*c_theo
    Ep = newton_G*Mh**2/Rh/2*f_c(c_theo)
    Ek = -Ep/2
    v = (2*Ek/Mh)**0.5
    mc = v*soliton_m_dev_v(particle_mass)

    s_h_eq = soliton_halo_equilibrium()
    
    return mc*s_h_eq*temp_from_c(current_redshift, Mh, particle_mass)



if __name__ == '__main__':


    ### load the command-line parameters to input your halo mass, particle mass, and redshift

    parser = argparse.ArgumentParser( description='Predicting the core-halo mass relation in fuzzy dark matter (FDM)' )

    parser.add_argument( '--halo_mass', action='store', required=False, type=float, dest='halo_mass',
                        help='halo mass', default=1e12 )
    parser.add_argument( '--redshift', action='store', required=False,  type=float, dest='redshift',
                        help='redshift', default=0 )
    parser.add_argument( '--particle_mass', action='store', required=False,  type=float, dest='particle_mass',
                        help='paticle mass', default=2e-23 )

    args=parser.parse_args()

    halo_mass           = args.halo_mass
    current_redshift    = args.redshift
    particle_mass       = args.particle_mass
    
    ### set cosmology
    #set_cosmology('planck18') # you can change to other cos,ology
    print(cosmology.getCurrent().name)

    print(f"halo mass: {halo_mass:.2e}, redshift: {current_redshift:.2e}, particle mass: {particle_mass:.2e}")

    ### Calculate the revised core mass
    revised_c_FDM_Mc = revised_theo_c_FDM_Mc(current_redshift, halo_mass, particle_mass)
    print(f"Predicted core mass (this work) : {revised_c_FDM_Mc:.2e}")

    ### Calculate the Schive2014 core mass
    theo_Mc = theo_TH_Mc(current_redshift, halo_mass, particle_mass)
    print(f"Predicted core mass (Schive2014) : {theo_Mc:.2e}")