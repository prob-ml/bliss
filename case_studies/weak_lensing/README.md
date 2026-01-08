### Neural posterior estimation for tomographic field-level weak lensing inference
#### Tim White, Shreyas Chandrashekaran, Dingrui Tao, Camille Avestruz, and Jeffrey Regier
#### with assistance from Steve Fan and Tahseen Younus

In this case study, we use neural posterior estimation to infer tomographic shear and convergence maps from LSST-like images.

To train the encoder on DC2 images, run

```
nohup bliss -cp <path>/bliss/case_studies/weak_lensing/dc2 -cn config_train_npe.yaml mode=train &> train_on_dc2.out &
```

To train the encoder on descwl-shear-sims images, run

```
nohup bliss -cp <path>/bliss/case_studies/weak_lensing/descwl -cn config_train_npe.yaml mode=train &> train_on_descwl.out &
```

See `dc2/results`, `dc2/exploratory`, `descwl/results`, and `descwl/exploratory` for analysis notebooks and results.
