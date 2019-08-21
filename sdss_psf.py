from numpy import *

# needs fitsio version 0.9.12
import fitsio


# Reconstruct the SDSS model PSF from KL basis functions.
#   hdu: the psField hdu for the band you are looking at.
#      eg, for r-band:
#	     psfield = pyfits.open('psField-%06i-%i-%04i.fit' % (run,camcol,field))
#        bandnum = 'ugriz'.index('r')
#	     hdu = psfield[bandnum+1]
#
#   x,y can be scalars or 1-d numpy arrays.
# Return value:
#    if x,y are scalars: a PSF image
#    if x,y are arrays:  a list of PSF images
def psf_at_points(x, y, psf_fit_file):
    # psfield = fitsio.FITS('sdss_stage_dir/2583/2/136/psField-002583-2-0136.fit')
     # 'sdss_stage_dir/3900/6/269/psField-003900-6-0269.fit'
    psfield = fitsio.FITS(psf_fit_file)
    hdu = psfield[3]

    rtnscalar = isscalar(x) and isscalar(y)
    x = atleast_1d(x)
    y = atleast_1d(y)

    psf = hdu.read()

    psfimgs = None
    (outh, outw) = (None, None)

    # From the IDL docs:
    # http://photo.astro.princeton.edu/photoop_doc.html#SDSS_PSF_RECON
    #   acoeff_k = SUM_i{ SUM_j{ (0.001*ROWC)^i * (0.001*COLC)^j * C_k_ij } }
    #   psfimage = SUM_k{ acoeff_k * RROWS_k }
    for k in range(len(psf)):
        nrb = psf[k]["nrow_b"]
        ncb = psf[k]["ncol_b"]

        c = psf[k]["c"].reshape(5, 5)
        c = c[:nrb, :ncb]

        (gridi, gridj) = meshgrid(range(nrb), range(ncb))

        if psfimgs is None:
            psfimgs = [zeros_like(hdu["rrows"][k][0]) for xy in broadcast(x, y)]
            (outh, outw) = (hdu["rnrow"][k][0], hdu["rncol"][k][0])

        for i, (xi, yi) in enumerate(broadcast(x, y)):
            acoeff_k = sum(((0.001 * xi) ** gridi * (0.001 * yi) ** gridj * c))
            psfimgs[i] += acoeff_k * hdu["rrows"][k][0]

    psfimgs = [img.reshape((outh, outw)) for img in psfimgs]

    if rtnscalar:
        return psfimgs[0]
    return psfimgs
