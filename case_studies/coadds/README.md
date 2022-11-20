# Coadd Case Studies

Code that reproduces the results for ``Statistical Inference for Coadded Astronomical Images'', 
workshop paper accepted to the NeurIPS 2022 Workshop on Machine Learning and the Physical Sciences.

The following steps will reproduce the results shown in this paper: 

0. The `config/config.yaml` controls almost all input parameters reproducing the results. In particular the `seed` argument
up top controls the randomness for the data produced. The `seed=42` was used for the results of the neurips paper.

1. First generate training, validation, and testing dataset: 

```bash
python get_dataset.py
```

2. Then, train all the coadd models used for results:

```bash
./train.sh single $GPU
./train.sh coadd_10 $GPU
./train.sh coadd_25 $GPU
./train.sh coadd_50 $GPU
```

where `GPU` is a variable corrresponding to the index of the GPU you want to use to train the model.

3. Finally run the following to get all plots from the paper.

```bash
python get_results.py results.overwrite=True # generate cache containing results to plot
python get_results.py # generate plots from cache
```

The figures are saved as `.png` files in `./outputs/figs/{SEED}/` as default, where `SEED` is the value
of the seed at the top of the config file.

## Updates

- 11/20/22: Flux measurement comparison (and flux posterior calibration) now conditions on true detections allowing for star-by-star comparison between models. 

- 11/16/22: Results for the ML4Science Neurips 2022 submission can be reproduced at commit [7b9f7aa](https://github.com/prob-ml/bliss/commit/7b9f7aaabd88b512e4d42432166a8ce319329bd9).
