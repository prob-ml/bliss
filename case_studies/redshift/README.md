This redshift estimation project is consist of 4 parts:
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
Run all the code at /home/qiaozhih/bliss/case_studies/redshift/evaluation/dc2_plot.ipynb
