## Run DC2 Cataloging Experiments on Great Lake

### Prepare a Singularity Container

You can use the `bliss_machine.def` file in this directory to generate a container. Please ensure you can run sudo in your machine. Running this on virtual machine with Linux system is recommended. Note: **DO NOT** run this using WSL2 which will lead to strange behavior.

```bash
sudo singularity build bliss_machine.sif bliss_machine.def
```

For your reference, we build the container on:

* Windows 11 Home (x86; Intel CPU)
* VirtualBox 7.0.18 with Ubuntu 24.04 LTS Server
* Singularity 4.1.0 built from source code

You can find our pre-built container at [this link](https://drive.google.com/file/d/1cezddVVZAnofFy4PZB_aOUaIOkFDvpQB/view?usp=sharing).

### Run the Container on Great Lake

Transfer the `bliss_machine.sif` to your Great Lake home folder, and run it with singularity. We write two simple scripts to show how to run it interactively (`run_bliss_machine_interactively.sh`) or using `sbatch` (`run_bliss_machine.sbatch` and `run_bliss_machine.sh`). If you want to configure your own container for your experiment, I suggest you to look at the following items before running your code:

* The `runscript` part in `bliss_machine.def`: it can switch to a specific git branch and run self-defined command using poetry. (Probably in most cases, you don't need to rebuild container.)
* The `paths` override in `configs/full_train_config_great_lake_exp_xxx`: make sure your data is on Great Lake and the output path is what you expect.
* The `matmul_precision` in `configs/full_train_config_great_lake_exp_xxx`: it may be helpful to set this tag if you are using GPU with Tensor Cores (like NVIDIA A40)
* The detailed settings in `run_bliss_machine.sbatch` and `run_bliss_machine.sh`

### Run directly on Great Lake

You can also run bliss directly on Great Lake. We provide two scripts (`run_great_lake.sbatch` and `run_great_lake.sh`) to show how to do. Due to this, we will not maintain the scripts and code w.r.t container. However, we still keep the container here because it may be useful if we want to run on other HPC.

### If you want to use Container in Future

You may need to fix the bug in `runscript` part of `bliss_machine.def`. The permission of `bliss` folder may be inconsistent with the git repository and cause failure when checking out.
