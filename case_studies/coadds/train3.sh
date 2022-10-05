# ./main.py mode=train training.name='encoder'
export CUDA_VISIBLE_DEVICES="7"
for i in 6 7 8 9 10
do
    ./main.py mode=train training.name="coadd_encoder_saved_25_all" training.experiment="coadd_encoder_saved_25_all" datasets.saved_coadd_module.train_ds.coadd_name="coadd_25" datasets.saved_coadd_module.val_ds.coadd_name="coadd_25" 'training.dataset="${datasets.saved_coadd_module}"' \
    training.seed=$i training.n_epochs=20 training.trainer.check_val_every_n_epoch=1 datasets.saved_coadd_module.train_ds.epoch_size=10000 datasets.saved_coadd_module.val_ds.epoch_size=10000 \

    ./main.py mode=train training.name="coadd_encoder_saved_50_all" training.experiment="coadd_encoder_saved_50_all" datasets.saved_coadd_module.train_ds.coadd_name="coadd_50" datasets.saved_coadd_module.val_ds.coadd_name="coadd_50" 'training.dataset="${datasets.saved_coadd_module}"' \
    training.seed=$i training.n_epochs=20 training.trainer.check_val_every_n_epoch=1 datasets.saved_coadd_module.train_ds.epoch_size=10000 datasets.saved_coadd_module.val_ds.epoch_size=10000
done




