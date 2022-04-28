# Coadd Catalog 

To obtain the coadd catalog that is used as ground truth for some studies, the CASJobs [website](https://skyserver.sdss.org/CasJobs/SubmitJob.aspx)
must be used. This requires the creation of a (free) account. For example, the coadd catalog `coadd_catalog_94_1_12.fits` which corresponds to the 
(run, camcol, field) = (94, 1, 12) was created using the following query.


```SQL
declare @SATURATED bigint set @SATURATED=dbo.fPhotoFlags('SATURATED')
declare @EDGE bigint set @EDGE=dbo.fPhotoFlags('EDGE')
declare @NODEBLEND bigint set @NODEBLEND=dbo.fPhotoFlags('NODEBLEND')
declare @bad_flags bigint set @bad_flags=(@EDGE|@NODEBLEND)

select *
from (
  select
    objid, type, mode, rerun, run, camcol, field, flags,
    ra, dec, probpsf,psfmag_r, devmag_r, expmag_r, modelMag_r,
    fracdev_r, devab_r, expab_r, devphi_r, expphi_r, devrad_r, exprad_r,
    (flags & @SATURATED) as is_saturated
  into coadd_catalog_23
  from stripe82.photoobj
  where
    run in (106, 206) and
    ra between 336.5825 and 336.7461 and
    dec between -1.044 and -0.8189
) as tmp
where
  ((psfmag_r < 23 and probpsf = 1) 
   or (probpsf = 0 and (modelMag_r < 23)))
  and (flags & @bad_flags) = 0
  and (type = 3 or type = 6) 
  and mode = 1
```

where the `ra` and `dec` limits correspond to the limits of the frame. They can be obtained: 

```python
from bliss.datasets.sdss import SloanDigitalSkySurvey
sdss = SloanDigitalSkySurvey(sdss_dir="../data/sdss", run=94, camcol=1, fields=(12,), bands=(2,))
h, w = sdss[0]['image'].shape[1:]
wcs = sdss[0]['wcs'][0]
print(wcs.all_pix2world([0, w], [0, h], 0))
```

The cut on magnitude is `23`, which can be adjusted depending on how dim of objects you want to consider.
For more details take a look at this [wiki page](https://github.com/jeff-regier/Celeste.jl/wiki/About-SDSS-and-Stripe-82#how-to-get-ground-truth-data-for-stripe-82). 
