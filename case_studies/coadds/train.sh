export CUDA_VISIBLE_DEVICES=0
./main.py mode=train experiment='coadd_encoder' training.name=coadd_encoder_1
# ./main.py mode=train training.name="second_run" training.n_epochs=2000
