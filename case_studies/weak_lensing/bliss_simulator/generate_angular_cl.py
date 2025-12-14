import numpy as np

try:
    import pyccl as ccl
except ModuleNotFoundError as err:
    raise ModuleNotFoundError("Please install pyccl using pip: pip install pyccl") from err


def main():
    vanilla_cosmo = ccl.cosmology.CosmologyVanillaLCDM()

    z = np.linspace(0.0, 3.0, 512)
    i_lim = 26.0
    z0 = 0.0417 * i_lim - 0.744
    ngal = 46.0 * 100.31 * (i_lim - 25.0)
    pz = 1.0 / (2.0 * z0) * (z / z0) ** 2.0 * np.exp(-z / z0)
    dndz = ngal * pz

    lensing_tracer = ccl.WeakLensingTracer(vanilla_cosmo, dndz=(z, dndz))

    ell = np.arange(1, 100000)
    angular_cl = ccl.angular_cl(vanilla_cosmo, lensing_tracer, lensing_tracer, ell)

    np.save("angular_cl.npy", np.stack((ell, angular_cl)))


if __name__ == "__main__":
    main()
