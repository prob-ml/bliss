singularity overlay create --sparse --size 5120 ./bliss_machine_overlay.img
singularity run --overlay bliss_machine_overlay.img bliss_machine.sif yd/test_container "bliss -cp /home/container_files/bliss/case_studies/dc2_cataloging/run_great_lake/configs -cn full_train_config_great_lake_exp_xxx"
singularity shell --overlay bliss_machine_overlay.img bliss_machine.sif
