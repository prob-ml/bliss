#!/usr/bin/env python3
"""
CosmoDC2 Galaxy Cluster Renderer

This script creates realistic astronomical images of galaxy clusters from CosmoDC2.

What this does:
- Searches the CosmoDC2 cosmological simulation for massive galaxy clusters
- Creates high-resolution color images that look like real telescope observations
- Produces both scientific FITS files and visual PNG images
"""

import logging
import os
import re
from datetime import datetime
from typing import Dict, List

import galsim
from galsim import fits
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from GCRCatalogs.cosmodc2 import CosmoDC2GalaxyCatalog


class CosmoDC2GalaxyClusterRenderer:
    """
    Creates realistic astronomical images of galaxy clusters from simulation data.

    This class turns catalog simulations into images that look like real telescope
    observations.

    What it does:
    1. Searches simulation data for massive galaxy clusters (>10^14 solar masses)
    2. Gets detailed properties of each galaxy (brightness, shape, size)
    3. Renders realistic images including telescope effects (blurring, noise)
    4. Creates both scientific data files and color images for visualization

    Technical details:
    - Uses CosmoDC2: A state-of-the-art cosmological simulation
    - Mimics DES telescope: Produces images matching real Dark Energy Survey data
    - HEALPix organization: Sky divided into equal-area regions for efficient processing
    - Multi-band photometry: Creates color images using 4 different light filters (g,r,i,z)

    HEALPix pixels explained:
    The simulation divides the sky into numbered regions (pixels). Each pixel contains
    thousands of galaxies. More pixels = larger sky area = more clusters found, but
    slower processing.
    """

    def __init__(
        self, catalog_dir: str = "/nfs/turbo/lsa-regier/lsstdesc-public/dc2/cosmoDC2_v1.1.4"
    ):
        """
        Set up the cluster image renderer.

        Args:
            catalog_dir: Path to CosmoDC2 simulation data files (HDF5 format)
                        Each file contains galaxies for one region of sky
        """
        self.catalog_dir = catalog_dir
        # Template for finding simulation files - each file covers one sky region
        self.catalog_filename_template = "z_{}_{}.step_all.healpix_{}.hdf5"

        # Create directory where final images will be saved
        self.output_dir = "output_ori"
        os.makedirs(self.output_dir, exist_ok=True)

        # Cosmological parameters used in the simulation
        # H0: Hubble constant (universe expansion rate)
        # Om0: Matter density (how much matter vs dark energy)
        # Ob0: Baryon density (ordinary matter like stars and planets)
        self.cosmology = {"H0": 71.0, "Om0": 0.2648, "Ob0": 0.0448}

        # Find all available sky regions in the simulation data
        self.available_pixels = self.get_available_healpix_pixels()
        logging.info(f"Discovered {len(self.available_pixels)} HEALPix pixels in {catalog_dir}")

    def get_available_healpix_pixels(self) -> List[int]:
        """
        Find all sky regions available in the simulation data.

        The simulation divides the sky into numbered regions (HEALPix pixels).
        Each region is stored in a separate file. This method finds all available
        regions by looking at the filenames.

        Think of it like: if you had photos of different neighborhoods, this would
        list all the neighborhood numbers you have photos for.

        Returns:
            List of region numbers (integers) that contain simulation data
        """
        files = os.listdir(self.catalog_dir)
        pixels = set()

        # Look through all files and extract region numbers from filenames
        for file in files:
            if file.startswith("z_") and file.endswith(".hdf5"):
                # Extract the pixel number from filename like "...healpix_9556.hdf5"
                match = re.search(r"healpix_(\d+)\.hdf5", file)
                if match:
                    pixels.add(int(match.group(1)))

        return sorted(list(pixels))

    def _create_catalog_for_healpix(self, healpix_id: int):
        """
        Load simulation data for one specific sky region.

        This creates a connection to the simulation data file for one region of sky.
        Each region contains thousands of galaxies with detailed properties like
        position, brightness, size, and shape.

        Args:
            healpix_id: Region number to load (corresponds to a specific area of sky)

        Returns:
            CosmoDC2GalaxyCatalog: Interface to query galaxy data in this region
        """
        logging.info(f"Loading catalog for HEALPix pixel {healpix_id}")
        # Create interface to simulation data with proper cosmological parameters
        catalog = CosmoDC2GalaxyCatalog(
            catalog_root_dir=self.catalog_dir,
            catalog_filename_template=self.catalog_filename_template,
            healpix_pixels=[healpix_id],  # Load just this one sky region
            version="1.1.4",  # CosmoDC2 simulation version
            cosmology=self.cosmology,  # Universe parameters used in simulation
        )
        return catalog

    def find_detectable_clusters(self, catalog):
        """
        Find galaxy clusters that are actually detectable by DES.

        This applies realistic redshift-dependent mass detection limits based on
        DES observational constraints using optical red-sequence detection.

        Detection limits account for:
        1. Galaxy evolution (red sequence develops with time)
        2. Surface brightness dimming with redshift
        3. DES survey depth and sensitivity limits
        4. Completeness and purity requirements

        Args:
            catalog: Interface to simulation data for one sky region

        Yields:
            Dictionary for each DES-detectable cluster containing position, mass, etc.
        """

        # DES optical detection limits (more realistic, based on actual redMaPPer performance)
        def min_detectable_mass(z):
            """Return minimum detectable mass for DES at given redshift."""
            if z < 0.2:
                return 2e13  # Low redshift, very sensitive
            elif z < 0.4:
                return 3e13  # Good detection efficiency
            elif z < 0.6:
                return 5e13  # Intermediate redshift
            elif z < 0.8:
                return 1e14  # Higher redshift, still detectable
            elif z < 1.0:
                return 2e14  # High redshift, massive clusters only
            else:
                return 1e16  # Effectively undetectable

        max_redshift = 1.0  # DES redshift limit

        # Start with a broad mass cut to reduce data volume
        # Use the minimum possible mass across all redshifts for DES
        broad_min_mass = min_detectable_mass(0.0)
        filters = [f"halo_mass >= {broad_min_mass}", f"redshift <= {max_redshift}"]

        # Properties we want to extract for each galaxy in massive halos
        quantities = [
            "galaxy_id",  # Unique identifier for each galaxy
            "ra",  # Right Ascension (longitude on sky, like GPS coordinates)
            "dec",  # Declination (latitude on sky)
            "redshift",  # How far away the galaxy is (higher = farther)
            "halo_mass",  # Total mass of dark matter halo (includes dark matter)
            "halo_id",  # Unique identifier for each cluster
            "mag_r",  # Brightness in red light (lower numbers = brighter)
            "mag_g",  # Brightness in green light
            "mag_i",  # Brightness in near-infrared light
            "mag_z",  # Brightness in infrared light
            "size_minor_true",  # Galaxy size along minor axis (arcseconds)
            "size_true",  # Galaxy size along major axis (arcseconds)
        ]

        logging.info(f"Querying catalog for DES-detectable clusters (z <= {max_redshift:.1f})...")
        data = catalog.get_quantities(quantities, filters=filters)
        logging.info(f"Found {len(data['ra'])} total galaxies in potential clusters")

        # Group galaxies by their parent halo to identify unique clusters
        # Many galaxies can belong to the same cluster (same halo_id)
        logging.info("Grouping galaxies by halo to identify unique clusters...")
        halo_ids = data["halo_id"]
        _, first_indices = np.unique(halo_ids, return_index=True)

        logging.info(f"Found {len(first_indices)} potential clusters before detectability filtering")

        # First pass: Apply redshift-dependent mass cuts to determine detectability
        detectable_clusters = []
        detectable_count = 0
        rejected_count = 0

        logging.info("Filtering clusters by DES detectability limits...")
        for i, idx in enumerate(first_indices):
            # Get cluster properties
            cluster_z = data["redshift"][idx]
            cluster_mass = data["halo_mass"][idx]
            min_detectable_mass_for_z = min_detectable_mass(cluster_z)

            # Check if cluster is detectable given its redshift and mass
            if cluster_mass >= min_detectable_mass_for_z:
                detectable_count += 1

                # Store detectable cluster info for processing
                cluster = {
                    "ra": data["ra"][idx],  # Sky position (longitude)
                    "dec": data["dec"][idx],  # Sky position (latitude)
                    "redshift": data["redshift"][idx],  # Distance indicator
                    "halo_mass": data["halo_mass"][idx],  # Total cluster mass
                    "halo_id": data["halo_id"][idx],  # Unique cluster identifier
                    "central_mag_r": data["mag_r"][idx],  # Central galaxy brightness (red)
                    "central_mag_g": data["mag_g"][idx],  # Central galaxy brightness (green)
                    "central_mag_i": data["mag_i"][idx],  # Central galaxy brightness (near-IR)
                    "central_mag_z": data["mag_z"][idx],  # Central galaxy brightness (IR)
                }
                detectable_clusters.append(cluster)
            else:
                rejected_count += 1
                logging.debug(
                    f"Rejected cluster: z={cluster_z:.3f}, M={cluster_mass:.2e} M☉ "
                    f"(below threshold: {min_detectable_mass_for_z:.2e} M☉)"
                )

        logging.info(
            f"DES detectability: {detectable_count} detectable, "
            f"{rejected_count} too faint/distant (total: {detectable_count + rejected_count})"
        )

        # Second pass: Yield the detectable clusters for processing
        for cluster in detectable_clusters:
            yield cluster

    def get_cluster_members(self, catalog, cluster: Dict) -> Dict:
        """
        Get detailed properties of all galaxies in a specific cluster.

        Each cluster contains hundreds of galaxies. This method finds all galaxies
        that belong to the same dark matter halo (same gravitational group).
        We need detailed shape and brightness information to create realistic images.

        Args:
            catalog: Interface to simulation data for one sky region
            cluster: Cluster summary from find_clusters_by_mass

        Returns:
            Dictionary containing arrays of properties for all member galaxies:
            - Positions, brightnesses, sizes, shapes, ellipticities
            - Everything needed to render realistic galaxy images
        """
        # Find all galaxies belonging to this cluster's dark matter halo
        # Using halo membership is more accurate than just finding nearby galaxies
        filters = [f'halo_id == {cluster["halo_id"]}']

        # Detailed properties needed to create realistic galaxy images
        quantities = [
            "galaxy_id",  # Unique identifier for each galaxy
            "ra",  # Sky position (longitude)
            "dec",  # Sky position (latitude)
            "redshift",  # Distance
            "mag_r",
            "mag_g",
            "mag_i",
            "mag_z",  # Brightness in 4 color filters
            # Galaxy structure: most galaxies have disk + bulge components
            "size_disk_true",  # Size of rotating disk (like spiral arms)
            "size_bulge_true",  # Size of central bulge (old stars)
            "size_minor_disk_true",  # Disk thickness
            "size_minor_bulge_true",  # Bulge minor axis
            # Galaxy shapes: how elliptical (stretched) each component is
            "ellipticity_1_disk_true",  # Disk ellipticity parameter 1
            "ellipticity_2_disk_true",  # Disk ellipticity parameter 2
            "ellipticity_1_bulge_true",  # Bulge ellipticity parameter 1
            "ellipticity_2_bulge_true",  # Bulge ellipticity parameter 2
            # What fraction of light comes from bulge vs disk
            "bulge_to_total_ratio_i",  # 0 = pure disk, 1 = pure bulge
        ]

        logging.info(f"Querying member galaxies for halo {cluster['halo_id']}...")
        data = catalog.get_quantities(quantities, filters=filters)

        logging.info(f"Found {len(data['ra'])} member galaxies for cluster")
        return data

    def process_all_healpix(self):
        """
        Process all available HEALPix pixels one by one,
        finding and rendering DES-detectable clusters as they're found.

        Saves both FITS files and PNG images for each cluster found.
        """
        total_clusters_found = 0

        logging.info(f"Starting pixel-by-pixel processing of {len(self.available_pixels)} pixels")

        for pixel_idx, healpix_pixel in enumerate(self.available_pixels):
            logging.info(
                f"Processing HEALPix Pixel {healpix_pixel} "
                f"({pixel_idx + 1}/{len(self.available_pixels)})"
            )

            # Create catalog for this healpix
            healpix_catalog = self._create_catalog_for_healpix(healpix_pixel)

            # Find detectable clusters in this healpix
            healpix_clusters_found = 0
            for cluster in self.find_detectable_clusters(healpix_catalog):
                healpix_clusters_found += 1
                total_clusters_found += 1

                logging.info(
                    f"DES-detectable cluster {total_clusters_found} (Pixel {healpix_pixel}) - "
                    f"RA: {cluster['ra']:.4f}°, Dec: {cluster['dec']:.4f}°"
                )
                logging.info(
                    f"Redshift: {cluster['redshift']:.4f}, "
                    f"Halo Mass: {cluster['halo_mass']:.2e} M☉, ID: {cluster['halo_id']}"
                )

                # Get cluster members
                members = self.get_cluster_members(healpix_catalog, cluster)
                logging.info(f"Member galaxies: {len(members['ra'])}")

                # Calculate basic statistics
                bright_members = sum(1 for mag in members["mag_r"] if mag < 25.0)
                logging.info(f"Bright members (r < 25): {bright_members}")

                # Render cluster
                band_images = self.render_cluster(cluster, members)

                # Save individual band images as FITS with DES calibration metadata
                logging.info("Saving FITS files with comprehensive headers...")
                self.save_as_fits(band_images, cluster, healpix_pixel, members)

                # Create and save color PNG
                color_filename = os.path.join(
                    self.output_dir,
                    f"cluster_halo_{cluster['halo_id']}_healpix_{healpix_pixel}.png",
                )
                self.save_as_png(band_images, color_filename)
                logging.info(f"Color image saved as {color_filename}")

                # Add separator after each cluster completion
                logging.info("=" * 80)

            if healpix_clusters_found == 0:
                logging.info(f"No clusters found in HEALPix pixel {healpix_pixel}")
            else:
                logging.info(
                    f"Found {healpix_clusters_found} clusters in HEALPix pixel {healpix_pixel}"
                )

        logging.info(
            f"Processing Complete - Total clusters found: {total_clusters_found} "
            f"across {len(self.available_pixels)} healpix"
        )
        logging.info(
            f"HEALPix-by-HEALPix processing complete. Found {total_clusters_found} clusters total."
        )

    def render_cluster(
        self, cluster: Dict, members: Dict, pixel_scale: float = 0.263, padding_factor: float = 1.5
    ) -> Dict[str, galsim.Image]:
        """
        Create realistic telescope images of a galaxy cluster.

        This is the main image creation function. It takes galaxy properties from the
        simulation and creates images that look like real telescope observations,
        including all the effects that telescopes see (blurring, noise, etc.).

        The process:
        1. Calculate how big the image needs to be to fit all galaxies
        2. Create separate images in 4 color filters (like taking photos with
           different colored glasses)
        3. For each galaxy, create a realistic shape and add it to the images
        4. Add telescope effects: atmospheric blurring, detector noise, sky background

        Args:
            cluster: Summary info about the cluster (position, mass, etc.)
            members: Detailed properties of all galaxies in the cluster
            pixel_scale: How much sky each pixel represents (0.263 arcsec = DES telescope)
            padding_factor: How much extra space to add around cluster (1.5 = 50% padding)

        Returns:
            Dictionary with 4 astronomical images (g, r, i, z bands)
            Each image contains realistic galaxy shapes with telescope effects
        """
        logging.info(f"Rendering cluster with {len(members['ra'])} members")

        # Set the image center to the cluster center
        center_ra = cluster["ra"]  # Sky longitude of cluster center
        center_dec = cluster["dec"]  # Sky latitude of cluster center

        # Convert galaxy positions from sky coordinates to image coordinates
        # Sky coordinates are in degrees, image coordinates in arcseconds
        # Factor of 3600 converts degrees to arcseconds (3600 arcsec = 1 degree)
        # Cosine correction needed because longitude lines converge at poles
        dx_arcsec = (members["ra"] - center_ra) * np.cos(np.radians(center_dec)) * 3600
        dy_arcsec = (members["dec"] - center_dec) * 3600

        # Figure out how big the image needs to be to contain all galaxies
        max_extent_x = max(abs(dx_arcsec.max()), abs(dx_arcsec.min()))
        max_extent_y = max(abs(dy_arcsec.max()), abs(dy_arcsec.min()))
        max_extent = max(max_extent_x, max_extent_y)

        # Add padding around the cluster so galaxies aren't at the edge
        field_size_arcsec = max_extent * padding_factor * 2  # *2 for both sides
        field_size_arcsec = max(field_size_arcsec, 60.0)  # At least 1 arcminute wide

        # Convert sky size to pixel size using telescope's pixel scale
        image_size = int(field_size_arcsec / pixel_scale)

        # Keep image size reasonable: at least 200 pixels, at most 5000 pixels
        # Make sure it's even (helps with image centering)
        image_size = max(200, min(5000, image_size + (image_size % 2)))

        logging.info(
            f"Cluster extent: {max_extent:.1f} arcsec, using {image_size}x{image_size} pixels "
            f"({field_size_arcsec:.1f}x{field_size_arcsec:.1f} arcsec)"
        )

        # Create separate images for each color filter (like taking 4 photos
        # with different colored glass)
        # g = blue light, r = red light, i = near-infrared, z = infrared
        bands = ["g", "r", "i", "z"]  # These match the Dark Energy Survey telescope filters
        band_images = {}

        for band in bands:
            # Create blank image filled with zeros
            band_images[band] = galsim.Image(image_size, image_size, scale=pixel_scale)

        # Create realistic DES Point Spread Function (PSF) for each band
        # Balance realism with computational efficiency

        # DES band-dependent seeing (atmospheric seeing varies with wavelength)
        band_fwhm = {"g": 0.92, "r": 0.88, "i": 0.85, "z": 0.83}  # arcsec, blue=worse seeing

        # Create band-specific PSFs with manageable computational requirements
        band_psfs = {}

        for band in bands:
            # Use Moffat profile: more realistic than Gaussian, computationally efficient
            # Moffat profiles have extended wings like real PSFs but render faster than Airy
            fwhm = band_fwhm[band]
            beta = 2.5  # Moffat beta parameter (2.5 is typical for good seeing)

            # Create the PSF with truncation to avoid huge FFTs
            psf = galsim.Moffat(fwhm=fwhm, beta=beta, trunc=4 * fwhm)

            # Add slight pixel-level smoothing for detector effects
            pixel_response = galsim.Pixel(scale=pixel_scale, flux=0.98)
            combined_psf = galsim.Convolve([psf, pixel_response])

            band_psfs[band] = combined_psf

            # Log the actual achieved FWHM
            actual_fwhm = combined_psf.calculateFWHM()
            logging.info(
                f'{band}-band PSF: Moffat β=2.5, target={fwhm:.2f}", actual={actual_fwhm:.2f}" FWHM'
            )

        # Add each galaxy to all 4 color images
        logging.info(f"Rendering {len(members['ra'])} galaxies in {len(bands)} bands...")

        for i, _ in enumerate(members["ra"]):
            for band in bands:
                # Render this galaxy in this color band with band-specific PSF
                self._render_galaxy(
                    band_images[band],
                    members,
                    i,
                    dx_arcsec,
                    dy_arcsec,
                    image_size,
                    pixel_scale,
                    band_psfs[band],
                    band,
                )

            # Log galaxy info once per galaxy (after all bands processed)
            logging.info(
                f"Galaxy {i+1} (ID {members['galaxy_id'][i]}): "
                f"mag_g={members['mag_g'][i]:.2f}, mag_r={members['mag_r'][i]:.2f}, "
                f"mag_i={members['mag_i'][i]:.2f}, mag_z={members['mag_z'][i]:.2f}"
            )

        # Add realistic DES coadd effects to make images look like real observations

        # Sky background: Even "empty" sky has some light (atmosphere, distant stars)
        # Different colors have different background brightness
        sky_mags_per_arcsec2 = {
            "g": 22.0,
            "r": 21.2,
            "i": 20.5,
            "z": 19.6,
        }  # DES sky brightness (mag/arcsec²)
        # Convert surface brightness (mag/arcsec²) to flux per pixel
        sky_levels = {
            band_name: self._surface_brightness_to_flux_per_pixel(
                sky_mags_per_arcsec2[band_name], band_name, pixel_scale
            )
            for band_name in bands
        }

        # DES coadd noise: Much lower than single exposures due to stacking ~50-100 exposures
        # DES Y6 coadd 5-sigma detection limits (deeper than single exposures)
        noise_mags_five_sigma = {"g": 25.8, "r": 25.2, "i": 24.6, "z": 23.8}  # DES coadd depths
        # Convert 5-sigma limits to 1-sigma: divide by 5, then account for coadd stacking
        noise_sigmas = {
            band_name: self._mag_to_flux(noise_mags_five_sigma[band_name], band_name) / 5.0
            for band_name in bands
        }  # 1-sigma noise level for realistic coadds

        for band in bands:
            # Add uniform sky background to every pixel
            band_images[band] += sky_levels[band]

            # Add random noise that varies pixel to pixel (like TV static)
            noise = galsim.GaussianNoise(sigma=noise_sigmas[band])
            band_images[band].addNoise(noise)

        logging.info("Cluster rendering complete")
        return band_images

    def _mag_to_flux(self, magnitude: float, band: str) -> float:
        """
        Convert astronomical magnitude to physical flux units.

        Magnitudes are how astronomers measure brightness, but they're backwards:
        - Lower magnitude = brighter object
        - Higher magnitude = fainter object
        - Each unit difference = 2.5x brighter/fainter

        Examples:
        - Sun: magnitude -27 (extremely bright)
        - Full moon: magnitude -13
        - Naked eye limit: magnitude +6
        - Hubble limit: magnitude +30 (extremely faint)

        This converts to nanomaggies, the standard unit used by modern surveys.

        Args:
            magnitude: Brightness in astronomical magnitudes (lower = brighter)
            band: Color filter ('g'=blue, 'r'=red, 'i'=near-IR, 'z'=IR)

        Returns:
            Flux in nanomaggies (physical brightness units used by DES survey)
        """
        # Zeropoints: calibration constants that convert magnitudes to physical units
        # These values are from the Dark Energy Survey Year 6 data release
        # Each color filter has a different zeropoint due to detector sensitivity
        des_zeropoints = {
            "g": 25.087,  # Blue light filter
            "r": 24.818,  # Red light filter
            "i": 24.406,  # Near-infrared filter
            "z": 23.885,  # Infrared filter
        }

        if band not in des_zeropoints:
            raise ValueError(f"Unknown band: {band}. Available: {list(des_zeropoints.keys())}")

        zeropoint = des_zeropoints[band]

        # Standard astronomical formula to convert magnitude to flux
        # The 2.5 comes from the definition of magnitudes (base-10 logarithm)
        flux_nmgy = 10 ** ((zeropoint - magnitude) / 2.5)

        return flux_nmgy

    def _surface_brightness_to_flux_per_pixel(
        self, mag_per_arcsec2: float, band: str, pixel_scale: float
    ) -> float:
        """
        Convert surface brightness (mag/arcsec²) to flux per pixel (nanomaggies).

        Args:
            mag_per_arcsec2: Surface brightness in magnitudes per square arcsecond
            band: Color filter ('g', 'r', 'i', 'z')
            pixel_scale: Pixel scale in arcsec/pixel

        Returns:
            Flux per pixel in nanomaggies
        """
        # Use the same zeropoints as point sources - this is correct for DES
        # Surface brightness and point source magnitudes use the same zeropoint system
        des_zeropoints = {
            "g": 25.087,  # Blue light filter
            "r": 24.818,  # Red light filter
            "i": 24.406,  # Near-infrared filter
            "z": 23.885,  # Infrared filter
        }

        if band not in des_zeropoints:
            raise ValueError(f"Unknown band: {band}. Available: {list(des_zeropoints.keys())}")

        zeropoint = des_zeropoints[band]
        pixel_area_arcsec2 = pixel_scale**2

        # Convert surface brightness to flux per square arcsecond, then scale by pixel area
        # mag/arcsec² -> nanomaggies/arcsec² -> nanomaggies/pixel
        flux_per_arcsec2 = 10 ** ((zeropoint - mag_per_arcsec2) / 2.5)
        flux_per_pixel = flux_per_arcsec2 * pixel_area_arcsec2

        return flux_per_pixel

    def _render_galaxy(
        self, full_image, members, i, dx_arcsec, dy_arcsec, image_size, pixel_scale, psf, band
    ):
        """
        Render a single galaxy onto the full image.

        Args:
            full_image: GalSim Image to render onto
            members: Member galaxy data dictionary
            i: Galaxy index
            dx_arcsec: X offset in arcseconds
            dy_arcsec: Y offset in arcseconds
            image_size: Size of the full image
            pixel_scale: Pixel scale in arcsec/pixel
            psf: PSF object
            band: Photometric band ('g', 'r', 'i')
        """
        # Use pre-calculated positions
        dx = dx_arcsec[i]
        dy = dy_arcsec[i]

        # Skip if outside image bounds
        if abs(dx) > image_size * pixel_scale / 2 or abs(dy) > image_size * pixel_scale / 2:
            return

        # Create galaxy profile using actual CosmoDC2 morphology data
        flux = self._mag_to_flux(members[f"mag_{band}"][i], band)

        # Get bulge-to-total ratio to determine dominant component
        bt_ratio = members["bulge_to_total_ratio_i"][i]

        # Get disk and bulge parameters
        disk_radius = max(0.1, members["size_disk_true"][i])
        bulge_radius = max(0.1, members["size_bulge_true"][i])

        # Get ellipticities for each component
        e1_disk = members["ellipticity_1_disk_true"][i]
        e2_disk = members["ellipticity_2_disk_true"][i]
        e1_bulge = members["ellipticity_1_bulge_true"][i]
        e2_bulge = members["ellipticity_2_bulge_true"][i]

        # Create bulge and disk components with their fluxes
        disk_flux = flux * (1 - bt_ratio)
        bulge_flux = flux * bt_ratio

        # Create realistic galaxy profile based on astronomical observations
        # Handle disk and bulge components uniformly - create components if they have flux
        components = []

        # Create disk component if it has flux
        if disk_flux > 0:
            disk = galsim.Exponential(half_light_radius=disk_radius, flux=disk_flux)
            if abs(e1_disk) > 0 or abs(e2_disk) > 0:
                disk = disk.shear(e1=e1_disk, e2=e2_disk)
            components.append(disk)

        # Create bulge component if it has flux
        if bulge_flux > 0:
            bulge = galsim.DeVaucouleurs(half_light_radius=bulge_radius, flux=bulge_flux)
            if abs(e1_bulge) > 0 or abs(e2_bulge) > 0:
                bulge = bulge.shear(e1=e1_bulge, e2=e2_bulge)
            components.append(bulge)

        # Combine components or create fallback
        if len(components) > 1:
            galaxy = galsim.Add(components)  # Multiple components: disk + bulge
        elif len(components) == 1:
            galaxy = components[0]  # Single component: pure disk or pure bulge
        else:
            # Fallback for unusual cases: create simple exponential galaxy
            galaxy = galsim.Exponential(half_light_radius=0.5, flux=flux)

        # Convolve with PSF
        final_profile = galsim.Convolve([galaxy, psf])


        # Draw galaxy directly onto the full image at the correct position
        # GalSim coordinate system: image center is at (image_size+1)/2
        center_x = (image_size + 1) / 2.0
        center_y = (image_size + 1) / 2.0

        # Convert arcsec offset to pixel offset
        x_pixel = center_x + dx / pixel_scale
        y_pixel = center_y + dy / pixel_scale

        final_profile.drawImage(
            full_image,
            add_to_image=True,
            offset=(x_pixel - int(x_pixel), y_pixel - int(y_pixel)),
            center=galsim.PositionI(int(x_pixel), int(y_pixel)),
            method="auto",
        )


    def save_as_png(self, band_images: Dict[str, galsim.Image], filename: str):
        """
        Save rendered multi-band images as background-subtracted color PNG
        file with DES-style normalization.

        Creates background-subtracted color image using all 4 DES bands (g,r,i,z):
        - Red channel: i+z bands (longer wavelengths)
        - Green channel: r+i bands (intermediate)
        - Blue channel: g band (shorter wavelength)
        - Sky background is subtracted to highlight galaxy structure

        Args:
            band_images: Dictionary with band names as keys and GalSim Image objects as values
            filename: Output PNG filename
        """
        # Extract band arrays
        g_array = band_images["g"].array
        r_array = band_images["r"].array
        i_array = band_images["i"].array
        z_array = band_images["z"].array

        # Background subtraction for each band
        # Subtract sky background to highlight galaxy structure (in nanomaggies)
        pixel_scale = 0.263  # arcsec/pixel
        sky_mags_per_arcsec2 = {
            "g": 22.0,
            "r": 21.2,
            "i": 20.5,
            "z": 19.6,
        }  # DES sky background in mag/arcsec²
        band_names = ["g", "r", "i", "z"]
        sky_levels = {
            band_key: self._surface_brightness_to_flux_per_pixel(
                sky_mags_per_arcsec2[band_key], band_key, pixel_scale
            )
            for band_key in band_names
        }

        # Subtract sky background from each band
        g_sub = g_array - sky_levels["g"]
        r_sub = r_array - sky_levels["r"]
        i_sub = i_array - sky_levels["i"]
        z_sub = z_array - sky_levels["z"]

        # Simple approach: just use percentile-based scaling
        def simple_scale(data, low_percentile=1, high_percentile=99.5):
            """Simple percentile-based scaling."""
            low = np.percentile(data, low_percentile)
            high = np.percentile(data, high_percentile)
            scaled = (data - low) / (high - low)
            return np.clip(scaled, 0, 1)

        # Create RGB channels directly
        red_channel = simple_scale(i_sub + 0.3 * z_sub)  # i+z for red
        green_channel = simple_scale(r_sub)              # r for green
        blue_channel = simple_scale(g_sub)               # g for blue

        # Stack into RGB image
        rgb_image = np.dstack([red_channel, green_channel, blue_channel])

        # Save the RGB image
        pil_image = Image.fromarray((rgb_image * 255).astype(np.uint8))
        pil_image.save(filename, dpi=(150, 150))

        logging.info(f"Background-subtracted color PNG saved as {filename}")

    def save_as_fits(
        self,
        band_images: Dict[str, galsim.Image],
        cluster: Dict,
        healpix_pixel: int,
        members: Dict = None,
    ):
        """
        Save rendered multi-band images as FITS files with comprehensive DES headers.

        Args:
            band_images: Dictionary with band names as keys and GalSim Image objects as values
            cluster: Cluster properties dictionary
            healpix_pixel: HEALPix pixel number for filename
            members: Member galaxy data dictionary (optional, for NMEMBERS header)
        """
        for band, image in band_images.items():
            fits_filename = os.path.join(
                self.output_dir,
                f"cluster_halo_{cluster['halo_id']}_healpix_{healpix_pixel}_{band}.fits",
            )

            # Get image properties for header
            image_size = image.array.shape[0]  # Image is square
            pixel_scale = 0.263  # DES pixel scale
            band_fwhm = {"g": 0.92, "r": 0.88, "i": 0.85, "z": 0.83}  # Band-dependent seeing
            psf_fwhm = band_fwhm[band]  # This band's PSF FWHM

            # DES calibration constants
            des_zeropoints = {"g": 25.087, "r": 24.818, "i": 24.406, "z": 23.885}
            sky_mags_per_arcsec2 = {
                "g": 22.0,
                "r": 21.2,
                "i": 20.5,
                "z": 19.6,
            }  # DES sky background in mag/arcsec²
            # DES Y6 coadd depths
            noise_mags_five_sigma = {"g": 25.8, "r": 25.2, "i": 24.6, "z": 23.8}
            band_names = ["g", "r", "i", "z"]
            sky_levels = {
                band_key: self._surface_brightness_to_flux_per_pixel(
                    sky_mags_per_arcsec2[band_key], band_key, pixel_scale
                )
                for band_key in band_names
            }
            # Convert 5-sigma limits to 1-sigma for realistic coadd noise levels
            noise_sigmas = {
                band_key: self._mag_to_flux(noise_mags_five_sigma[band_key], band_key) / 5.0
                for band_key in band_names
            }

            # Create background-subtracted image
            bg_subtracted_image = galsim.Image(image.array - sky_levels[band], scale=pixel_scale)

            # Create comprehensive FITS header matching DES coadd standards
            header = fits.FitsHeader()

            # Basic FITS keywords
            header["SIMPLE"] = True
            header["BITPIX"] = -32  # 32-bit floating point
            header["NAXIS"] = 2  # 2D image
            header["NAXIS1"] = image_size
            header["NAXIS2"] = image_size
            header["EXTEND"] = True

            # DES survey identification
            header["SURVEY"] = "DES"
            header["ORIGIN"] = "CosmoDC2 Simulation"
            header["TELESCOP"] = "Blanco"
            header["INSTRUME"] = "DECam"
            header["OBSERVER"] = "CosmoDC2"
            header["OBJECT"] = f"Galaxy_Cluster_{cluster['halo_id']}"

            # Filter and photometric calibration
            header["FILTER"] = band.upper()
            header["BAND"] = band
            header["MAGZERO"] = des_zeropoints[band]
            header["MAGZP"] = des_zeropoints[band]
            header["MAGZPUNC"] = 0.01  # Typical DES zeropoint uncertainty
            header["FLXSCALE"] = 1.0

            # Flux units and calibration
            header["BUNIT"] = "nanomaggy"
            header["FLXUNIT"] = "nmgy"
            header["UNITS"] = "nanomaggy"

            # World Coordinate System (WCS)
            header["CTYPE1"] = "RA---TAN"
            header["CTYPE2"] = "DEC--TAN"
            header["CRPIX1"] = (image_size + 1) / 2.0  # Reference pixel (center)
            header["CRPIX2"] = (image_size + 1) / 2.0
            header["CRVAL1"] = cluster["ra"]  # RA at reference pixel
            header["CRVAL2"] = cluster["dec"]  # Dec at reference pixel
            header["CD1_1"] = (
                -pixel_scale / 3600.0
            )  # degrees per pixel (negative for standard orientation)
            header["CD1_2"] = 0.0
            header["CD2_1"] = 0.0
            header["CD2_2"] = pixel_scale / 3600.0  # degrees per pixel
            header["RADESYS"] = "ICRS"  # Coordinate system
            header["EQUINOX"] = 2000.0

            # Image properties
            header["PIXSCALE"] = pixel_scale  # arcsec/pixel
            header["PIXAREA"] = pixel_scale**2  # arcsec^2/pixel
            header["SATURATE"] = 1e6  # Saturation level (high for simulation)
            header["GAIN"] = 1.0  # e-/ADU (set to 1 for nanomaggy units)
            header["RDNOISE"] = noise_sigmas[band]  # Read noise in image units

            # PSF information
            header["SEEING"] = psf_fwhm  # arcsec
            header["PSF_FWHM"] = psf_fwhm  # arcsec
            header["AIRMASS"] = 1.2  # Typical zenith angle

            # Sky background (subtracted from image)
            header["SKYBRITE"] = sky_mags_per_arcsec2[band]  # mag/arcsec^2 (subtracted)
            header["SKYSIGMA"] = noise_sigmas[band]  # Sky noise in image units
            header["SKYLEVEL"] = 0.0  # Sky level (subtracted, now zero)
            header["SKYSUBT"] = True  # Sky background has been subtracted

            # Cluster properties
            header["HALO_ID"] = cluster["halo_id"]
            header["HALOMASS"] = cluster["halo_mass"]  # Solar masses
            header["REDSHIFT"] = cluster["redshift"]
            header["RA_CLUS"] = cluster["ra"]  # Cluster center RA
            header["DEC_CLUS"] = cluster["dec"]  # Cluster center Dec
            if members is not None:
                header["NMEMBERS"] = len(members["ra"])  # Number of member galaxies

            # Processing information
            header["DATE"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            header["SOFTWARE"] = "GalSim + CosmoDC2"
            header["VERSION"] = "1.0"
            header["SIMTYPE"] = "CosmoDC2 v1.1.4"

            # Comments
            header["COMMENT"] = f"Background-subtracted DES {band}-band image of galaxy cluster"
            header["COMMENT"] = f"Flux units: nanomaggies (zeropoint = {des_zeropoints[band]} mag)"
            header["COMMENT"] = f"Sky background subtracted (was {sky_levels[band]:.3f} nmgy/pix)"
            header["COMMENT"] = "Generated from CosmoDC2 cosmological simulation"
            header["COMMENT"] = "Realistic galaxy morphologies and DES observing conditions"
            header["HISTORY"] = "Created by CosmoDC2 Galaxy Cluster Renderer"

            # Attach header to background-subtracted image and write FITS file
            bg_subtracted_image.header = header
            bg_subtracted_image.write(fits_filename)
            logging.info(
                f"FITS {band}-band saved as {fits_filename} (background-subtracted nanomaggy units)"
            )


def main():
    """
    Main function: find and render DES-detectable galaxy clusters in the simulation.

    This will:
    1. Search all available sky regions for DES-detectable galaxy clusters
    2. Apply realistic redshift-dependent mass detection limits
    3. Create realistic telescope images of each detectable cluster
    4. Save both scientific FITS files and color PNG images
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Create the renderer and process all available data
    finder = CosmoDC2GalaxyClusterRenderer()

    # Process all detectable clusters and save both FITS and PNG files
    finder.process_all_healpix()


if __name__ == "__main__":
    main()
