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

# create a lock file to prevent multiple instances of this script from running
if { set -C; 2>/dev/null >$HOME/htr2hpc.lock; }; then
    trap "rm -f $HOME/htr2hpc.lock" EXIT
else
    echo "Lock file exists... another instance of the script may already be running. Exiting."
    exit
fi

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

# htr2hpc conda environment is too large for default adroit home.
# move .conda to scratch and create a symlink, if one does not already exist
CONDA_SYM="/scratch/network/$USER/.conda"
if [ -L "$HOME/.conda" ] && [ $(readlink -f $HOME/.conda) = $CONDA_SYM ]; then
	echo ".conda already relocated to user scratch"
elif [ -L "$HOME/.conda" ]; then
	CONDA_SYM=$(readlink -f $HOME/.conda)
	echo ".conda has existing sym link $CONDA_SYM"
else 
	echo "Porting .conda to scratch and creating symlink in home"
	rsync -avu $HOME/.conda /scratch/network/$USER/
	rm -Rf $HOME/.conda
	if [ ! -d "/scratch/network/$USER/.conda" ]; then
		echo "Creating /scratch/network/$USER/.conda directory"
		mkdir /scratch/network/$USER/.conda
	fi
	cd $HOME && ln -s /scratch/network/$USER/.conda .conda
fi

# create conda environment named htr2hpc
conda_env_name=htr2hpc
module load anaconda3/2025.6
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
	conda create -y -n $conda_env_name python=3.11 pip
	conda activate $conda_env_name
	pip install -q kraken==6.0.3
	pip install -q git+https://github.com/Princeton-CDH/htr2hpc.git@develop#egg=htr2hpc
fi

htrworkingdir=/scratch/network/$USER/htr2hpc
# create working directory
if [ ! -d $htrworkingdir ]; then
	echo "Creating htr2hpc working directory in scratch: $htrworkingdir"
	mkdir $htrworkingdir
else
	echo "htr2hpc scratch working directory already exists: $htrworkingdir"
fi

echo "Setup complete!"
