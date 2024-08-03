Spatially Varying Backgrounds and PSFs Case Study
============================================

This case study contains code to reproduce the results and figures for the paper "Neural Posterior Estimation for Cataloging Astronomical Images with Spatially Varying Backgrounds and Point Spread Functions".

To run the experiments, run
```
sh run_experiments.sh
```

This will generate two synthetic datasets, train three models on the appropriate dataset, and run evaluation and generate figures similar to those seen in the paper. Note that the results may not be exact due to differences in the generated data.

There are three main config files, one for each model:
- `conf/single_field.yaml`
- `conf/psf_unaware.yaml`
- `conf/psf_aware.yaml`

You should modify the `paths/cached_data` in each file to point to where you would like to save the generated datasets.
