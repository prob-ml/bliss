### Neural posterior estimation of weak lensing shear and convergence
#### Tim White, Shreyas Chandrashekaran, Camille Avestruz, and Jeffrey Regier
#### with assistance from Dingrui Tao, Steve Fan, and Tahseen Younus

This case study aims to estimate weak lensing shear and convergence for the DC2 simulated sky survey. See `notebooks/dc2/manuscript` for our most recent results.

Some useful commands:

- Train `lensing_encoder` on DC2 images

```
nohup bliss -cp /home/twhit/bliss/case_studies/weak_lensing/ -cn lensing_config_dc2.yaml mode=train &> train_on_dc2.out &
```

- Generate synthetic images with shear and convergence, as specified in `lensing_prior`

```
nohup bliss -cp /home/twhit/bliss/case_studies/weak_lensing/ -cn lensing_config_simulator.yaml mode=generate &> generate_synthetic.out &
```

- Train `lensing_encoder` on synthetic images:

```
nohup bliss -cp /home/twhit/bliss/case_studies/weak_lensing/ -cn lensing_config_simulator.yaml mode=train &> train_on_synthetic.out &
```
