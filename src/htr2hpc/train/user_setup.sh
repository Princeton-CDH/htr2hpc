#!/bin/bash

# bash script to setup user account for htr2hpc pilot integration
# - adds ssh key to authorized keys
# - conda env setup
# - create htr2hpc working directory in scratch

# defaults
ssh_setup=true
reinstall_htr2hpc=false

# supported options:
#   --skip-ssh-setup
#   --reinstall-htr2hpc
for arg in "$@"; do
  if [[ "$arg" == "--skip-ssh-setup" ]]; then
    ssh_setup=false
  elif [[ "$arg" == "--reinstall-htr2hpc" ]]; then
	reinstall_htr2hpc=true
  fi
done

echo "Setting up your account for htr2hpc ...."
echo "This process may take five minutes or more on first run. Do not exit until the process completes."


# skip ssh setup if --skip-ssh-setup is specified
if $ssh_setup; then
	# ensure ssh directory exists
	if [ ! -d "$HOME/.ssh" ]; then
	    echo "Creating $HOME/.ssh directory"
	    mkdir ~/.ssh
	fi

	# add test-htr public key to authorized keys if not already present
	ssh_key='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJzoR8jstrofzFKVoiXSFP5jGw/WbXHxFyIaS5b4vSWC test-htr.lib.princeton.edu'
	if ! grep -q "$ssh_key" $HOME/.ssh/authorized_keys; then
	    echo "Adding htr2hpc ssh key to authorized keys"
	    echo $ssh_key >> ~/.ssh/authorized_keys
	else
	    echo "ssh key is already in authorized keys"
	fi
fi

# create conda environment named htr2hpc
conda_env_name=htr2hpc
module load anaconda3/2024.2
if { conda env list | grep $conda_env_name; } >/dev/null 2>&1; then
	echo "conda env $conda_env_name already exists"

	# when conda env already exists, if requested
	# uninstall and reinstall htr2hpc
	if $reinstall_htr2hpc; then
		echo "Reinstalling htr2hpc"
		conda activate $conda_env_name
		pip uninstall -q --yes htr2hpc
		pip install -q git+https://github.com/Princeton-CDH/htr2hpc.git@develop#egg=htr2hpc
	fi

else
	echo "Creating conda environment $conda_env_name and installing dependencies"
	cd /scratch/gpfs/rkoeser/htr2hpc_setup/kraken
	conda env create -f environment_cuda.yml -n $conda_env_name
	conda activate $conda_env_name
	pip install -q torchvision torch==2.1 torchaudio==2.1
	pip install -q git+https://github.com/Princeton-CDH/htr2hpc.git@develop#egg=htr2hpc
	# go back to previous directory
	cd -
fi

htrworkingdir=/scratch/gpfs/$USER/htr2hpc
# create working directory
if [ ! -d $htrworkingdir ]; then
	echo "Creating htr2hpc working directory in scratch: $htrworkingdir"
	mkdir $htrworkingdir
else
	echo "htr2hpc scratch working directory already exists: $htrworkingdir"
fi

echo "Setup complete! 🚀 🚃"
