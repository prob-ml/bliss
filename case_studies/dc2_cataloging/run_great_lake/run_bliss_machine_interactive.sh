salloc --job-name=interactive \
       --account=regier0 \
       --partition=spgpu \
       --nodes=1 \
       --ntasks=1 \
       --gpus-per-node=a40:1 \
       --cpus-per-gpu=16 \
       --mem-per-gpu=32GB \
       --time=02:00:00

module load singularity/4.1.2 cuda/12.1.1 cudnn/12.1-v8.9.0

EXP_POSTFIX="test"
EXP_CUR_PATH=`pwd`

mkdir -p ./output
mkdir -p ./overlay_imgs
mkdir -p /tmp_data/pduan/dc2local/

rsync -av --info=progress2 \
 -e "ssh -i ~/.ssh/jeffbox_great_lake_side_rsa" \
 pduan@deeplearning-01.stat.lsa.umich.edu:/data/scratch/dc2local/dc2_split_results.tar \
 /tmp_data/pduan/dc2local/
pv /tmp_data/pduan/dc2local/dc2_split_results.tar | tar -xf - -C /tmp_data/pduan/dc2local/

singularity overlay create --sparse --size 5120 "./overlay_imgs/bliss_machine_overlay_${EXP_POSTFIX}.img"
singularity run --nv \
 --bind /tmp_data/:/tmp_data/ \
 --overlay "./overlay_imgs/bliss_machine_overlay_${EXP_POSTFIX}.img" \
 bliss_machine.sif \
 yd/test_container \
 "bliss -cp /home/container_files/bliss/case_studies/dc2_cataloging -cn full_train_config_great_lake train.trainer.logger.version=great_lake_exp_${EXP_POSTFIX}"
