salloc --job-name=interactive \
       --account=regier0 \
       --partition=spgpu \
       --nodes=1 \
       --ntasks=1 \
       --gpus-per-node=a40:1 \
       --cpus-per-gpu=16 \
       --mem-per-gpu=32GB \
       --time=00:10:00

module load singularity/4.1.2 cuda/12.1.1 cudnn/12.1-v8.9.0
singularity overlay create --sparse --size 5120 "bliss_machine_overlay_test.img"
singularity run --nv --overlay bliss_machine_overlay_test.img bliss_machine.sif yd/test_container \
 "bliss -cp /home/container_files/bliss/case_studies/dc2_cataloging/run_great_lake/configs -cn full_train_config_great_lake_exp_06-16-1"
