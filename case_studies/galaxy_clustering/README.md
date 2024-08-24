# Inference on Galaxy Cluster Fields

Summer 2024

[Ishan Kapnadak](https://www.linkedin.com/in/ishan-kapnadak/), [Gabriel Patron](https://lsa.umich.edu/stats/people/phd-students/gapatron.html), [Prof. Jeffrey Regier](https://regier.stat.lsa.umich.edu/), [Prof. Camille Avestruz](https://cavestruz.github.io/).

Built on work done in Winter 2024 by [Li Shihang](https://www.linkedin.com/in/shihang-li-2b69251ba/), Zhao Yongxiang.

----------------------------------------------------------------------------------------------------------------------

## Generation of Data

The data generation routine proceeds through phases. The entire routine is conveniently wrapped into a single python script `data_gen.py` under the `data_generation` subdirectory, which draws its parameters from the Hydra configuration, located under `conf/config.yaml` under the `data_gen` key. These phases proceed as follows.

1. **Catalog Generation.** First, we sample semi-synthetic source catalogs with their relevant properties, which are stored as `.dat` files in the `data_dir/catalogs` subdirectory.
2. **Image Generation.** Then, we take in the aforementioned source catalogs and use GalSim to render them as images, which are stored as `.fits` files (one for each band) in the `data_dir/images` subdirectory.
3. **File Datum Generation.** Finally, we convert the full source catalogs generated in phase 1 into tile catalogs, stack them up with their corresponding images, and store these objects as `.pt` files (which is what the encoder ultimately uses) in the `data_dir/file_data` subdirectory.

The following parameters can be set within the configuration file `config.yaml`.
1. `data_dir`: the path of the directory where generated data will be stored.
2. `image_size`: size of the image (pixels).
3. `tile_size`: size of tile to be used (pixels).
4. `nfiles`: number of files to be generated.
5. `n_catalogs_per_file`: number of catalogs to be stored in each file datum object.
6. `bands`: survey bands to be used (`["g", "r", "i", "z"]` for DES).
7. `min_flux_for_loss`: minimum flux for filtering.
8. `overwrite`: whether to overwrite existing files (only works for catalogs for now)


## Training the Encoder

Scripts related to the encoder can be found under the `encoder` subdirectory. We define our own `ClusterMembershipAccuracy` metric class under `metrics.py` that computes and logs (i) accuracy, (ii) precision, (iii) recall, and (iv) F1 score.

The individual components and the overall network used is described in `convnets.py`. We use *GroupNorm* in our features network since this works well with small batch sizes (we have a batch size of 1 or 2 given the large image size). The network downsamples successively according to the tile size.

The `encoder.py` file defines the custom encoder. The encoder parameters are again set within the Hydra configuration, under the `encoder` key under `conf/config.yaml`. The cached data simulator draws its parameters from the `cached_simulator` key under the same file, where one can specify the data directory, any training transforms to be applied, train-validation-test split, and batch size.

The encoder can be trained using the command (note that the path may need to be changed based on where you are running the command from -- I have used an absolute path for convenience). This command reads in our desired configuration file, and passes it on to BLISS. The outputs are logged in the `training_gc.out` file. GPU index can be specified beforehand by running the command `export CUDA_VISIBLE_DEVICES=${gpu_index}`.

```
nohup bliss -cp /home/kapnadak/bliss/case_studies/galaxy_clustering/conf/ --config-name config.yaml mode=train &> training_gc.out
```

The trained encoder and training logs are saved under `bliss_output` where one can find the TensorBoard event files (for plotting) as well as the best encoder under `checkpoints/best_encoder.ckpt`.

## Encoder Inference

After the encoder is appropriately trained, we can then test its performance against DES data by completing a full pass over the entirety of the survey. Configuration parameters can be specified in the config file located in `conf/config.yaml` under the `predict` section, and because they might vary from run to run, practitioners might find most useful modifying:

1. `cached_data_path`: the path of the dr2 data.
2. `batch_size`: size of the batches for each GPU.
3. `num_workers`: number of workers for the DataLoader.
4. `devices`: if input is of `int` type, number of GPU devices, and if input is a list of ints, then it specifies the GPU indices.
5. `output_dir`: number of catalogs to be stored in each file datum object.
6. `weight_save_path`: encoder weights location.

The **dr2 tiles** that make up DES data come as coadded bands of single-epoch exposures by band, totaling _20 GB_ of storage. Hence, to avoid unnecesary storage redundancy,
`DES_inference.py` builds just-in-time images in parallel and discards them after they are processed by the encoder. This process is specified in the `cached_dataset.py` module, under the `inference` subdirectory.
With all configuration parameters worked out for inference on DES, you can run the following command under the `galaxy_clustering` case study directory,:

```
python ./inference/DES_inference.py
```

The callback specification determines that the the outputs are saved at each batch end, so encoder outputs should be available as soon as the run begins processing images. For reference, a full run over DES typically takes 4-5 distributed on two 2080ti GPUs and 4 workers.

## Evaluating the Encoder

There are scripts under the `inference` subdirectory to run large-scale inference on the DES data. Since our encoder works on $2560 \times 2560$ images, we resort to unfold the DES images (which are $10,000 times 10,000$) into a $4 \times 4$ grid of $2560 \times 2560$ images using `torch`'s unfolding capabilities. The `inference_stats.py` script also counts the number of clusters we have predicted and computes metrics against redMaPPer (which we assume to be the ground truth). We only have redMaPPer predictions on the SVA1 footprint, so the metrics are restricted to this region. (`data_generation/SVA_map.py` and `data_generation/redmapper_groundtruth.py` are useful files for seeing how the ground truth results are logged.)

We can also visualize how our encoder works on individual DES tiles. In particular, `notebooks.SingleTileEvaluation.ipynb` does a good job of visualizing BLISS v/s redMaPPer predictions on individual DES tiles.
