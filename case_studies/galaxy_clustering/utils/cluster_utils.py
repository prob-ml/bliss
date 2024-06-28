import numpy as np
from astropy import constants as astroc
from astropy import units
from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology import cosmology
from colossus.halo import concentration, mass_defs

# Here you need to set your cosmology to the one used in the paper where M500 is given
# The baryon fraction and sigma8 are likely these params, perhaps double check

G_NEWTON = astroc.G.to(
    units.Mpc * units.km**2 / units.s**2 / units.solMass,
)  # Gravitational constant
H0 = 70 * units.km / units.s / units.Mpc  # Hubble constant
OM0 = 0.3  # Omega for matter
ODE = 0.7  # Omega for dark energy
OB0 = 0.049  # Baryon fraction
SIGMA8 = 0.81  # sigma8
NS = 0.95  # ?
M0 = (
    8 * 10**13 * units.solMass
)  # M0 for calculating number of clustered galaxies from virial mass (in solar masses)
N_BETA = 1.35  # beta for calculating number of clustered galaxies from virial mass


def hubble_parameter(z):
    """Hubble parameter at a given redshift.

    Args:
        z: redshift

    Returns:
        Hubble parameter at redshift z (in km/s/Mpc)
    """
    e_z = np.sqrt(ODE + OM0 * (1 + z) ** 3.0)
    return H0 * e_z


def critical_density(z):
    """Critical density at a given redshift.

    Args:
        z: redshift

    Returns:
        critical density at redshift z (in solMass/Mpc**3)
    """
    return 3 * (hubble_parameter(z) ** 2.0) / (8 * np.pi * G_NEWTON)


def m200_to_r200(m200, z):
    """Converting m200 to r200 at a given redshift.

    Args:
        m200: virial mass (in solMass)
        z: redshift

    Returns:
        r200 (virial radius in Mpc)
    """
    rho_z = critical_density(z)
    return (3 * m200 / (4 * np.pi * 200.0 * rho_z)) ** (1.0 / 3.0)


def m500_to_r500(m500, z):
    """Converting m500 to r500 at a given redshift.

    Args:
        m500: mass within r500 (in solMass)
        z: redshift

    Returns:
        r500 in Mpc
    """
    rho_z = critical_density(z)
    return (3 * m500 / (4 * np.pi * 500.0 * rho_z)) ** (1.0 / 3.0)


def m500_to_m200(m500, z):
    """Converting m500 to m200 at a given redshift.
    Assumes an Navarro-Frenk-White Profile

    Args:
        m500: mass within r500 (in solMass)
        z: redshift

    Returns:
        m200 (virial mass in solMass)
    """
    params = {"flat": True, "H0": H0.value, "Om0": OM0, "Ob0": OB0, "sigma8": SIGMA8, "ns": NS}
    cosmology.addCosmology("myCosmo", **params)
    cosmology.setCosmology("myCosmo")
    # This line converts M500 to c500, calibrated using extensive simulations and observations
    # See Duffy 2008 for more details
    c500 = concentration.concentration(m500, "200c", z, model="duffy08")
    # The next line converts M500 to M200 assuming an NFW profile and the concentration above
    return mass_defs.changeMassDefinition(m500, c500, z, "500c", "200c", profile="nfw")[0]


def m200_to_n200(m200):
    """Number of clustered galaxies for a given virial mass.

    Args:
        m200: virial mass (in solMass)

    Returns:
        n200 (number of clustered galaxies within R200)
    """
    return 20 * (m200 / M0) ** (1 / N_BETA)


def mag_to_flux(mag):
    """Converts magnitude to flux.

    Args:
        mag: magnitude

    Returns:
        flux
    """
    return 10 ** ((mag - 30) / -2.5)


def redshift_distribution(z, alpha, beta, z0):
    """Redshift distribution which follows the functional form in Chang et al (2013).
    Source: https://github.com/LSSTDESC/CLMM/blob/main/clmm/redshift/distributions.py

    Args:
        z: redshift
        alpha: exponent for z
        beta: exponent for z/z0
        z0: scaling parameter

    Returns:
        p(z): the probability density at the given redshift
    """
    return (z**alpha) * np.exp(-((z / z0) ** beta))


def angular_diameter_distance(z):
    """Get angular diameter distance at a given redshift.

    Args:
        z: redshift

    Returns:
        Da(z) in Mpc
    """
    cosmo = FlatLambdaCDM(H0=H0, Om0=OM0)
    return cosmo.angular_diameter_distance(z)
