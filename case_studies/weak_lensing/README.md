## Neural posterior estimation for field-level weak lensing inference
#### Tim White, Shreyas Chandrashekaran, Dingrui Tao, Camille Avestruz, and Jeffrey Regier, with assistance from Steve Fan and Tahseen Younus

In this case study, we use neural posterior estimation (NPE) to infer weak lensing shear and convergence from LSST-like images. We use NPE to (1) infer tomographic mass maps for the [DC2 Simulated Sky Survey](https://data.lsstdesc.org/doc/dc2_sim_sky_survey) and (2) infer constant shear from images generated with the [`descwl-shear-sims` package](https://github.com/timwhite0/descwl-shear-sims).

## DC2

### Generate catalog

```
nohup python -u case_studies/weak_lensing/dc2/generate_catalog.py &> generate_catalog.out &
```

### Generate mass maps and train MassMapEncoder

```
nohup bliss -cp <path>/bliss/case_studies/weak_lensing/dc2 -cn config_train_npe.yaml mode=train &> train_dc2.out &
```

### Notebooks

- **In `dc2/results`**: `credibleintervals.ipynb`, `posteriormeanmaps.ipynb`
- **In `dc2/exploratory`**: `dc2imageandmaps.ipynb`, `ellipticity.ipynb`, `galaxyproperties.ipynb`, `twopoint.ipynb`

## descwl-shear-sims

### Train ScalarShearEncoder

```
nohup bliss -cp <path>/bliss/case_studies/weak_lensing/descwl -cn config_train_npe.yaml mode=train &> train_descwl.out &
```

### Run [AnaCal](https://github.com/mr-superonion/AnaCal)

Configure settings in `descwl/config_run_anacal.yaml`.

```
nohup python -u case_studies/weak_lensing/descwl/run_anacal.py &> run_anacal.out &
```

### Notebooks

- **In `descwl/results`**: `compute_npe_credibleintervals.py`, `credibleintervals.ipynb`, `scatterplots.ipynb`
- **In `descwl/exploratory`**: `images.ipynb`
