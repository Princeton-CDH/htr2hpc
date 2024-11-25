# How to use `htr2hpc.train`

This submodule provides fucntionality for running HTR segmentation and transcription model training and fine-tuning tasks on an HPC system using content and models downloaded from an eScriptorium instance.

The `htr2hpc-train` script will download document parts and optionally a model file and then start a slurm job to run the appropriate training task.

## Setup

On the HPC system (i.e., della for Princeton), create a new python 3.11 conda environment named `htr2hpc` and install the current version of this software:

```sh
module load anaconda3/2024.6
wget https://github.com/mittagessen/kraken/raw/refs/heads/main/environment_cuda.yml
conda env create -f environment_cuda.yml -n htr2hpc
conda activate htr2hpc
pip install git+https://github.com/Princeton-CDH/htr2hpc.git@feature/export-and-train-noparsl#egg=htr2hpc
```

To tinker with htr2hpc settings, check out the code from github and install the local checkout as an editable installation:
```sh
git clone https://github.com/Princeton-CDH/htr2hpc.git
cd htr2hpc
git checkout feature/export-and-train-noparsl
pip install -e .
```

Change directory to your scratch space (e.g., `/scratch/gpfs/netid/` on della) or a subdirectory somewhere under it. Since the `htr2hpc-train` script monitors the slurm job, it is recommended to start a tmux session so that if you're disconnected the script will keep running and you can reconnect.

Set your eScriptorium API token as an environment variable:
```sh
export ESCRIPTORIUM_API_TOKEN=#####
```

To start a segmentation training task, specify:
 - the mode (segmentation)
 - the base url for the eScriptorium instance
 - a directory name where the script should download files (directory must not exist)
 - an eScriptorium document id
 - an optional model id

For example:
```sh
htr2hpc-train segmentation https://test-htr.lib.princeton.edu/ segtrain_doc2 --document 30 --model 3
```

To reuse previously downloaded data, add the `--existing-data` flag.

The output from the slurm job will be placed in the working directory you specified when running the htr2hpc-train command, in a file named `segtrain_####.out` where #### is the slurm job id.