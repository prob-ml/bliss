# ./main.py mode=train training.name='encoder'
export CUDA_VISIBLE_DEVICES="7"
for i in 1 2 3 4 5
do
    ./main.py mode=train training.name="single_encoder" training.experiment="single_encoder" datasets.saved_coadd_module.train_ds.coadd_name="single" datasets.saved_coadd_module.val_ds.coadd_name="single" 'training.dataset="${datasets.saved_coadd_module}"' training.seed=$i training.n_epochs=301

    ./main.py mode=train training.name="coadd_encoder_saved_10" training.experiment="coadd_encoder_saved10" datasets.saved_coadd_module.train_ds.coadd_name="coadd_10" datasets.saved_coadd_module.val_ds.coadd_name="coadd_10" 'training.dataset="${datasets.saved_coadd_module}"' training.seed=$i training.n_epochs=301

    ./main.py mode=train training.name="coadd_encoder_saved_25" training.experiment="coadd_encoder_saved25" datasets.saved_coadd_module.train_ds.coadd_name="coadd_25" datasets.saved_coadd_module.val_ds.coadd_name="coadd_25" 'training.dataset="${datasets.saved_coadd_module}"' training.seed=$i training.n_epochs=301

    ./main.py mode=train training.name="coadd_encoder_saved_50" training.experiment="coadd_encoder_saved50" datasets.saved_coadd_module.train_ds.coadd_name="coadd_50" datasets.saved_coadd_module.val_ds.coadd_name="coadd_50" 'training.dataset="${datasets.saved_coadd_module}"' training.seed=$i training.n_epochs=301
done
