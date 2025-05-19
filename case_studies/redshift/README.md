# BLISS-PZ - BLISS For Photo-z Prediction

#### Running BLISS-PZ on DC2

Modify the config file `redshift_xxxxxx.yaml` as follows:
1. Change `paths.root` to your `bliss` folder and `paths.data_dir` to a directory where you're happy to have all data artifacts and checkpoints stored.
2. Make the `OUT_DIR` variable in `runner.sh` to the location `{paths.data_dir}/training_logs` for logging.
3. Modify `paths.dc2` to the location of `dc2` on your system.

To produce the results from `BLISS-PZ`, run `runner.sh` (you made need to make this an executable, `chmod +x runner.sh` from within this directory).

```
./runner.sh
```

Comment out portions of `runner.sh` that you don't want to run. The current script has options to run BLISS-PZ with i) a discretized variational distribution, ii) a single univariate Gaussian (continuous), iii) a mixture density network (MDN), iv) a B-spline variational distribution. 

The current workflow is as follows:

1) Run data preparation (UNDER CONSTRUCTION...use batches currently located in `/data/scratch/declan/redshift/dc2/cached_dc2`)
2) Launch timestamped run by executing `./runner.sh`. Choose device and MDN/B-Spline version by modifying `CUDA_VISIBLE_DEVICES` and commenting out relevant commands.
3) At completion of training, checkpointed weights are located in `{paths.data_dir}/checkpoints`. 
4) In `case_studies/redshift/evaluation`, modify `xxxx_eval.yaml` with the correct `ckpt_dir` that you just generated (TODO: make accessing latest timestamp more automatic)
5) Run `case_studies/redshift/evaluation/evaluate_xxxx.py`, which loads the checkpoint from your supplied `ckpt_dir`and creates `.parquet` files of predictions on the test set. At present, the parquet files contain, for each matched object, the true redshift, true magnitudes, predicted redshift (which is the mode of the variational distribution) and NLL (of the true redshift under the predicted variational posterior). The `.parquet` files are stored in `{paths.data_dir}/plots`.
6) Use the resulting parquet files to compute test metrics of your choice (e.g., L2, L1 or NLL for now).
7) Currently, `case_studies/redshift/evaluation/plots_bliss.py` contains some example code used to produce plots. Most of this is outdated and needs to be replaced, but it can be used as a template. 



<!-- The runner bash script launches programs sequentially: first data prep, then two different runs of BLISS, followed by RAIL. Thereafter, plots are produced. For your use case it may be better to run different parts of the runner script on their own. Take a look at the script and comment out the relevant parts if you need. -->




<!-- This redshift estimation project is consist of 4 parts:
1. Estimate photo-z using neural network (training data is GT mag and redshift)
2. Estimate photo-z using bliss directly from image.
3. Estimate photo-z using lsst + rail pipeline (model from LSST)
4. Estimate photo-z using lsst + pretrained neural network from 1.

There are a few things need to do to make sure you can do evaluation(make plottings) on these four parts(I suggest you first go to the fourth part to see if you miss some key parts for evaluation)

1. Training neural network
Skip this step if you already have pretrained network. My pretrained network is saved at /data/scratch/qiaozhih/DC2_redshift_training/DC2_redshift_only_bin_allmetrics/checkpoints/encoder_0.182845.ckpt
`
./preprocess_dataset.sh
./train.sh
`

2. Train bliss
run /home/qiaozhih/bliss/case_studies/redshift/redshift_from_img/train.sh
You can modify config at /home/qiaozhih/bliss/case_studies/redshift/redshift_from_img/full_train_config_redshift.yaml

3. Train rail
All training code can be found at /home/qiaozhih/bliss/case_studies/redshift/evaluation/rail/RAIL_estimation_demo.ipynb. Make sure you install rail from and you must make sure you are using the corresponding env from rail instead of the bliss.

4. Evaluate & make plot
Run all the code at /home/qiaozhih/bliss/case_studies/redshift/evaluation/dc2_plot.ipynb -->
