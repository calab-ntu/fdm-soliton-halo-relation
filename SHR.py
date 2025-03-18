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
kpc2km               = constants.KPC*1e-5                                    # km
hbar                 = constants.H*0.5/np.pi/constants.KPC**2/constants.MSUN # kpc^2 msun/s
eV_c2                = constants.EV/constants.C**2/constants.MSUN            # Msun
newton_G             = constants.G/kpc2km**2                                 # (kpc^3)/(s^2*Msun)


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
        Schive2014b eq. 6 & 7 https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.261302

        Args:
            redshift (float)  : Redshift.
            halo_mass (float) : Halo mass in Msun.
            m22 (float)       : Particle mass in 1e-22 eV.

        Returns:
            ms (float)        : Theoretical soliton mass for the halo in Msun.
            rs (float)        : Theoretical soliton radius for the halo in kpc.
            peak_dens (float) : Peak density of the soliton in Msun/kpc**3.
            Rh (float)        : Halo radius in kpc.
        """

        current_time_a = redshift_to_a(current_redshift)
        zeta           = get_zeta(current_redshift, self.omega_M0)
        zeta_0         = get_zeta(0, self.omega_M0)
        Mmin0          = 4.4e7*m22**(-3/2)            # Msun

        ms             = 0.25*current_time_a**(-0.5)*(zeta/zeta_0)**(1/6)*(Mh/Mmin0)**(1/3)*Mmin0
        rs             = 1.6/m22*current_time_a**(0.5)*(zeta/zeta_0)**(-1/6)*(Mh*1e-9)**(-1/3)
        peak_dens      = soliton_dens(0, rs, m22)

        Rh            = (3*Mh/(4*np.pi*zeta*(self.background_density_0/current_time_a**3)))**(1/3)

        return ms, rs, peak_dens, Rh

    def revised_theo_c_FDM_Ms(self,current_redshift, Mh, m22):
        """
        Calculates the revised theoretical soliton mass for a halo in FDM.
        Liao2024 https://arxiv.org/abs/2412.09908

        Args:
            redshift (float)  : Redshift.
            halo_mass (float) : Halo mass in Msun.
            m22 (float)       : Particle mass in 1e-22 eV.

        Returns:
            ms (float)        : Revised theoretical soliton mass for the halo in Msun.
            rs (float)        : Revised theoretical soliton radius for the halo in kpc.
            peak_dens (float) : Peak density of the soliton in Msun/kpc**3.
            Rh (float)        : Halo radius in kpc.
            NFW_Rs (float)    : NFW scale radius in kpc.
            c_theo (float)    : Theoretical halo concentration for FDM.
            c_CDM (float)     : Prediction of halo concentration in CDM model.
            beta (float)      : Nonisothermality parameter.
        """

        current_time_a = redshift_to_a(current_redshift)
        c_CDM          = self.concentration_para_CDM(Mh, current_redshift)
        c_theo         = self.concentration_para_FDM(Mh, current_redshift, m22)
        zeta           = get_zeta(current_redshift, self.omega_M0)

        # Compute the virialized halo radius Rh (see get_zeta function).
        Rh             = (3*Mh/(4*np.pi*zeta*(self.background_density_0/current_time_a**3)))**(1/3) # kpc
        NFW_Rs         = Rh/c_theo

        # Define key physical constants related to halo dynamics and nonisothermality:
        alpha          = 2**(-0.5)                       # (1+(⟨v_h⟩/⟨w_h⟩)**2)**-0.5
        beta           = self.nonisothermality(c_theo)   # w_{h, in}/⟨w_h⟩
        gamma          = 0.8935555051894757              # <w_s>/w_{h, in}

        # Calculate the potential energy Ep of the halo, assuming a NFW density profile.
        Ep             = get_Ep(Mh, Rh, c_theo, 'NFW')   # Msun*kpc**2/s**2
        # Use Ep to find the halo velocity vh. 1/2 Mh*vh^2 = Ek = -1/2 Ep.
        vh             = (-Ep/Mh)**0.5                   # kpc/s
        # Get soliton thermal velocity ws by considering alpha (energy equipartition), beta (nonisothermality), and gamma (thermal equilibrium)
        ws             = vh*alpha*beta*gamma             # kpc/s

        # Soliton mass `ms` is calculated using the simplified formula, based on scaling relations.
        ms             = 3.15e8*ws*kpc2km/100*m22**-1

        # You can also use the `soliton_m_div_v` function directly.
        # ms             = soliton_m_div_v(m22)*ws

        rs             = soliton_m_mul_r(m22)/ms
        peak_dens      = soliton_dens(0, rs, m22)

        return ms, rs, peak_dens, Rh, NFW_Rs, c_theo, c_CDM, beta

    def revised_theo_c_FDM_Rs(self,current_redshift, Rh, m22):
        """
        Calculates the revised theoretical soliton radius for a halo in FDM.
        
        Args:
            redshift (float)  : Redshift.
            Rh (float)        : Halo radius in kpc.
            m22 (float)       : Particle mass in 1e-22 eV.
        
        Returns:
            ms (float)        : Revised theoretical soliton mass for the halo in Msun.
            rs (float)        : Revised theoretical soliton radius for the halo in kpc.
            Mh (float)        : Halo mass in Msun.
            NFW_Rs (float)    : NFW scale radius in kpc.
            c_theo (float)    : Theoretical halo concentration for FDM.
        """

        current_time_a = redshift_to_a(current_redshift)
        zeta           = get_zeta(current_redshift, self.omega_M0)
        Mh = 4*np.pi/3*Rh**3*self.background_density_0/current_time_a*zeta

        ms, rs, peak_dens, Rh, NFW_Rs, c_theo, c_CDM, beta = self.revised_theo_c_FDM_Ms(current_redshift, Mh, m22)

        return ms, rs, peak_dens, Mh, NFW_Rs, c_theo, c_CDM, beta

    def concentration_para_CDM(self, halo_mass, redshift):
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

        c_CDM = concentration.concentration(halo_mass*self.h, 'vir', redshift, model = 'ishiyama21',halo_sample ='relaxed')

        return c_CDM

    def concentration_para_FDM(self, halo_mass, redshift, m22):
        """
        Calculates the halo concentration for FDM with Laroche suppression.

        Args:
            halo_mass (float) : Halo mass in Msun.
            redshift (float)  : Redshift.
            m22 (float)       : Particle mass in 1e-22 eV.

        Returns:
            c_FDM (float)     : Prediction of halo concentration for FDM.
        """

        def FDM_supress_laroche(M, m22):
            """
            Calculates the LaRoche suppression factor for FDM using corrected values from Kawai (2024).
            Kawai2024 identified typos in the a, b, c values of LaRoche (2022) and refitted them.
            Laroche2022 https://academic.oup.com/mnras/article/517/2/1867/6711699
            Kawai2024 https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.023519

            Args:
                halo_mass (float) : Halo mass in Msun.
                m22 (float)       : Particle mass in 1e-22 eV.

            Returns:
                F (float)         : LaRoche suppression factor.
            """

            a , b, c = 5.496, -1.648, -0.417
            x        = M/half_mode_mass(m22)
            F        = (1+a*x**b)**c

            return F

        c_CDM     = self.concentration_para_CDM(halo_mass, redshift)
        F_supress = FDM_supress_laroche(halo_mass, m22)
        c_FDM     = c_CDM*F_supress

        return c_FDM

    def nonisothermality(self, c_FDM):
        """
        Calculates the ratio of the inner-halo thermal velocity to the average thermal velocity of the entire halo from the halo concentration for FDM by empirical fitting.

        Args:
            c_FDM (float)            : The concentration parameter of FDM halo.

        Returns:
            nonisothermality (float) : The ratio of the inner-halo thermal velocity to the average thermal velocity of the entire halo.
        """

        nonisothermality = 0.281*np.log10(c_FDM)+1.05

        return nonisothermality


def half_mode_mass(m22):
    """
    Returun half mode mass. (Mass below which small structures are suppressed due to FDM wave-like behavior)
    Schive2016 eq. 6 https://iopscience.iop.org/article/10.3847/0004-637X/818/1/89

    Args:
        m22 (float)      : Particle mass in 1e-22 eV.

    Returns:
        h_m_mass (float) : Half mode mass in Msun.
    """

    h_m_mass = 3.8e10*m22**(-4/3)

    return h_m_mass

def get_Ep(Mh, Rh, c, type):
    """
    Returun halo potential energy from density distribution.
    Assume the halo is spherical symmertry and ignore the mass outside the halo radius.

    Args:
        Mh (float)   : Halo mass in Msun.
        Rh (float)   : Halo radius in kpc.
        c (float)    : Halo concentration parameter. (only when type = 'NFW)
        type (string) : The halo density distribution. Currently support 'NFW' and 'Top Hat'.

    Returns:
        Ep (float)   : Potential energy of halo in Msun*kpc^2/s^2.
    """

    def f_c(c):
        """
        Coefficient of NFW potential energy as a function of concentration parameter.
        This ignores contributions outside the halo radius.
        \phi_{NFW} = (-4 \pi G \rho_0 R_s^3) / r \times \ln (1+r/R_s) + 4\pi G \rho_0R_s^3/R_{vir} \times({c}/{1+c})
        dM_{NFW}   = ( 4 \pi r^2 \rho_0) / [ (r/R_s) \times (1+r/R_s)^2 ] dr
        E_p  = 1/2 \int_0^{R_{vir}} \phi_{NFW} dM_{NFW}

        Args:
            c (float) : Concnetration parameter.

        Returns:
            Coefficient.
        """
        return (2*c*(1+c)*np.log(1+c)-2*c**2-c**3)/((1+c)*np.log(1+c)-c)**2

    def f_c_with_outside(c):
        """
        Coefficient of NFW potential energy as a function of concentration parameter.
        This include contributions outside the halo radius.
        \phi_{NFW} = (-4 \pi G \rho_0 R_s^3) / r \times \ln (1+r/R_s)
        dM_{NFW}   = ( 4 \pi r^2 \rho_0) / [ (r/R_s) \times (1+r/R_s)^2 ] dr
        E_p  = 1/2 \int_0^{R_{vir}} \phi_{NFW} dM_{NFW}
        Bar 2018 https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.083027

        Args:
            c (float) : Concnetration parameter.

        Returns:
            Coefficient.
        """
        return c*(1+c)*(np.log(1+c)-c)/((1+c)*np.log(1+c)-c)**2

    if type == 'NFW':
        Ep = newton_G*Mh**2/Rh/2*f_c(c)
    elif type == 'Top Hat':
        Ep = newton_G*Mh**2/Rh*-0.6
    else:
        raise ValueError(f"Unsupported model type '{type}'. Supported types are 'NFW' and 'Top Hat'.")

    return Ep

def soliton_dens(x, core_radius, m22):
    """
    Calculates the soliton density profile in physical frame. The core radius marks where density falls to half its peak.
    Schive2014a Supplement eq. 4 https://arxiv.org/abs/1406.6586

    Args:
        x (float)           : radius in kpc
        core_radius (float) : Core radius in kpc.
        m22 (float)         : Particle mass in 1e-22 eV.

    Returns:
        density (float)     : Soliton density at the given radius in Msun/kpc**3.
    """

    density = ((1.945*(m22/10**-1)**-2*(core_radius**-4))/((1 + 9.06*10**-2*(x/core_radius)**2)**8))*10**9

    return density

def grad_soliton(x, core_radius, m22):
    """
    Calculates the gradient of soliton density profile in physical frame.

    Args:
        x (float)           : Radius in kpc.
        core_radius (float) : Core radius in kpc.
        m22 (float)         : Particle mass in 1e-22 eV.

    Returns:
        dens_gradient (float) : The gradient of soliton density at the given radius in Msun/kpc^4.
    """

    dens_gradient = (1.945*(m22/10**-1)**-2*(core_radius**-4)*(-9.06*10**-2*16*x/core_radius**2)/((1 + 9.06*10**-2*(x/core_radius)**2)**9))*10**9

    return dens_gradient

def soliton_shell_mass(r, core_radius, m22):
    """
    Calculates the shell mass at radius r.

    Args:
        r (float)           : radius in kpc.
        core_radius (float) : Core radius in kpc.
        m22 (float)         : Particle mass in 1e-22 eV.

    Returns:
        mass (float) : Shell mass of the soliton at radius r in  Msun/kpc.
    """

    mass = 4*np.pi*r**2*soliton_dens(r, core_radius, m22)

    return mass

def soliton_mass(m22, core_radius, enclose_r = 1):
    """
    Calculates the soliton mass enclosed within a given radius. The default radius is the core radius.

    Args:
        m22 (float)         : Particle mass in 1e-22 eV.
        core_radius (float) : Core radius in kpc.
        enclose_r (float)   : The enclosed radius in core radius. Default is 1.

    Returns:
        mass (float)           : Soliton mass enclosed in Msun.
    """

    mass = quad(lambda r: soliton_shell_mass(r,core_radius, m22), 0, core_radius*enclose_r)[0]

    return mass

def soliton_m_mul_r(m22, enclose_r = 1):
    """
    Calculates the soliton mass multiplied by its enclosed radius in physical frame.
    This value is inversely proportional to the square of a given particle's mass.

    Args:
        m22 (float)            : Particle mass in 1e-22 eV.
        enclose_r (float)      : The enclosed radius, with a default value of the core radius.

    Returns:
        m_mul_r (float)        : Soliton's mass * radius in Msun*kpc
    """

    core_radius = 4 # kpc. Any core_radius value can be used to evaluate the constant.

    ms_enclose = soliton_mass(m22, core_radius, enclose_r) # Msun
    m_mul_r    = ms_enclose*core_radius

    return m_mul_r

def soliton_m_div_v(m22, enclose_r = 3.3):
    """
    Calculates the soliton mass divided by its enclosed average velocity in physical frame.
    This value is proportional to a given particle mass.

    Args:
        m22 (float)            : Particle mass in 1e-22 eV.
        enclose_r (float)      : The enclosed radius, with a default value of 3.3 times the core radius. This value typically encloses 95% of the energy.

    Returns:
        m_divided_by_v (float) : Soliton's mass / velcoity in Msun/kpc/s
    """

    core_radius = 4 # kpc. Any core_radius value can be used to evaluate the constant.

    def Ek_func(r, m22):
        """
        Calculates the kinetic energy Ek in physical frame.

        Args:
            r (float)   : radius in kpc.
            m22 (float) : Particle mass in 1e-22 eV.

        Returns:
            Ek (float)  : Kinetic energy in this shell in Msun*kpc/s^2
        """

        v  = 0.5/soliton_dens(r, core_radius, m22)*(hbar/(m22*1e-22*eV_c2))*grad_soliton(r, core_radius, m22)
        Ek = 0.5*soliton_shell_mass(r, core_radius, m22)*v**2

        return Ek

    ms_enclose     = soliton_mass(m22, core_radius, enclose_r)                       # Msun
    Eks            = quad(lambda r: Ek_func(r, m22), 0, core_radius*enclose_r)[0]    # Msun*kpc^2/s^2

    vs             = (2*Eks/ms_enclose)**0.5                                         # kpc/s
    ms             = soliton_mass(m22, core_radius, 1)                               # Msun
    m_divided_by_v = ms/vs

    return m_divided_by_v


def get_zeta(redshift, omega_M0):
    """
    Calculates the zeta parameter for a spherical cluster model.
    Bryan1998 eq. 6 https://iopscience.iop.org/article/10.1086/305262
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


if __name__ == '__main__':


    ### load the command-line parameters to input your halo mass, particle mass, and redshift

    parser = argparse.ArgumentParser(description='Predicting the soliton-halo mass relation in fuzzy dark matter (FDM)')
    
    # Create a mutually exclusive group for halo_mass and halo_radius
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-hm', '--halo_mass',       action='store', type=float, dest='halo_mass',
                        help='halo mass (Msun)', default=None)
    group.add_argument('-hr', '--halo_radius',     action='store', type=float, dest='halo_radius',
                        help='halo radius (kpc)', default=None)

    parser.add_argument('-z',   '--redshift',      action='store', type=float, dest='redshift',
                        help='redshift',              default=0)
    parser.add_argument('-m22', '--m22',           action='store', type=float, dest='m22',
                        help='paticle mass (1e-22 eV)', default=2e-1)

    args=parser.parse_args()

    # Check if at least one of halo_mass or halo_radius is provided
    if args.halo_mass is None and args.halo_radius is None:
        print("No halo mass or halo radius is provided. Use default halo mass 1e12 Msun.")
        args.halo_mass = 1e12


    Mh               = args.halo_mass
    Rh               = args.halo_radius
    current_redshift = args.redshift
    m22              = args.m22

    ### set cosmology
    # Initialize a SHR_calculator class. You can change to other cosmology
    shr_calculator = SHR_calculator('planck18')
    print(f"\n{'='*50}\nCosmological Model: {shr_calculator.cosmo.name}\n{'='*50}")

    print("\nInput Parameters:")


    ### Calculate the revised soliton mass by given halo mass
    if args.halo_mass is not None:
        print(f"{'Halo Mass':<20}: {Mh:.2e} Msun")
        Ms, Rs, peak_dens, Rh, NFW_scale_radius, c_theo, c_CDM, beta = shr_calculator.revised_theo_c_FDM_Ms(current_redshift, Mh, m22)
    ### Calculate the revised soliton mass by given halo radius
    else:
        print(f"{'Halo Radius':<20}: {Rh:.2e} kpc")
        Ms, Rs, peak_dens, Mh, NFW_scale_radius, c_theo, c_CDM, beta = shr_calculator.revised_theo_c_FDM_Rs(current_redshift, Rh, m22)

    print(f"{'Redshift':<20}: {current_redshift:.2e}")
    print(f"{'m22':<20}: {m22:.2e}")
    print(f"\n{'-'*50}\nLiao2024 Predictions\n{'-'*50}")

    print("\nSoliton Properties:")
    print(f"{'Mass':<20}: {Ms:.2e} Msun")
    print(f"{'Radius':<20}: {Rs:.2e} kpc")
    print(f"{'Peak Density':<20}: {peak_dens:.2e} Msun/kpc^3")

    print("\nHalo Properties:")
    print(f"{'Mass':<20}: {Mh:.2e} Msun")
    print(f"{'Radius':<20}: {Rh:.2e} kpc")
    print(f"{'NFW Scale Radius':<20}: {NFW_scale_radius:.2e} kpc")
    print(f"{'Concentration FDM':<20}: {c_theo:.2e}")
    print(f"{'Concentration CDM':<20}: {c_CDM:.2e}")
    print(f"{'nonisothermality':<20}: {beta:.2e}")
    print(f"\n{'='*50}")

    ### Calculate the Schive2014 soliton mass
    Ms, Rs, peak_dens, Rh = shr_calculator.theo_TH_Ms(current_redshift, Mh, m22)
    print(f"\n{'-'*50}\nSchive2014 Predictions\n{'-'*50}")

    print("\nSoliton Properties:")
    print(f"{'Mass':<20}: {Ms:.2e} Msun")
    print(f"{'Radius':<20}: {Rs:.2e} kpc")
    print(f"{'Peak Density':<20}: {peak_dens:.2e} Msun/kpc^3")

    print("\nHalo Properties:")
    print(f"{'Mass':<20}: {Mh:.2e} Msun")
    print(f"{'Radius':<20}: {Rh:.2e} kpc")
    print(f"\n{'='*50}")

