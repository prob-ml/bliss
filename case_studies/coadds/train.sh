# ./main.py mode=train training.name='encoder'
export TRAIN_EPOCHS=30
export EPOCH_SIZE=10000
export MODEL_NAME=$1 #options single / coadd_XYZ
export CUDA_VISIBLE_DEVICES=$2
for i in 1 2 3 4 5
do
    ./main.py mode=train training.name="${MODEL_NAME}_encoder" \
    training.experiment="${MODEL_NAME}_encoder" \
    datasets.saved_coadd_module.train_ds.coadd_name="${MODEL_NAME}" \
    datasets.saved_coadd_module.val_ds.coadd_name="${MODEL_NAME}" \
    training.seed=$i training.n_epochs=$TRAIN_EPOCHS \
    training.trainer.check_val_every_n_epoch=1 \
    datasets.saved_coadd_module.train_ds.epoch_size=$EPOCH_SIZE \
    datasets.saved_coadd_module.val_ds.epoch_size=$EPOCH_SIZE \
    'paths.output="${paths.project}/output/training/"'
done
