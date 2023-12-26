#!/bin/bash

#Install gh


#type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)

#type -p gh >/dev/null && echo Found gh || (curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
#&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
#&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
#&& sudo apt update \
#&& sudo apt install gh -y)


#cd ${HOME}
#if [ ! -d ${HOME}/venv ]; then
#	python3 -m virtualenv venv
#	source ${HOME}/venv/bin/activate
#	pip install -U pip setuptools wheel
#fi
#pip install --upgrade torch huggingface transformers transformers[torch] trl hf_transfer
#pip list
#       dd	&& rm -rf ${HOME}/venv

cd ${HOME}
disks=$(lsblk -d | grep disk | awk '{print $1}' | sort)
mountpoint=0
setup_hf=0
for d in ${disks}
do
	echo Checking disk ${d}
	inuse=$(mount -v | grep ${d} | sort | wc -l)
	if [ "${inuse}" == "0" ]; then 
	       	echo "Disk ${d} not mounted. Checking for fylesystem"
		fstype=$(blkid -o value -s TYPE /dev/nvme1n1)
		if [ "${fstype}" == "" ]; then
			echo "Disk ${d} not mounted and not formatted"
			sudo mkfs.xfs -f /dev/${d}
		else
			echo "Disk ${d} not mounted and formatted for ${fstype}"
		fi
		[ ! -d /mnt/disk${mountpoint} ] && sudo mkdir /mnt/disk${mountpoint}
		sudo mount /dev/${d} /mnt/disk${mountpoint}
		sudo chown -R ${USER} /mnt/disk${mountpoint}
		echo Disk ${d} mounted on  /mnt/disk${mountpoint}
		if [ "${setup_hf}" == "0" ]; then	
			cache_hf_dir=${HOME}/.cache/huggingface
			echo Setting up hf on new disk ${d} /mnt/disk${mountpoint}/huggingface
			[ ! -d /mnt/disk${mountpoint}/huggingface ] && mkdir /mnt/disk${mountpoint}/huggingface
			[ -d ${HOME}/.cache/huggingface ] && cd ${HOME}/.cache/huggingface && (tar czvf - . | (cd /mnt/disk${mountpoint}/huggingface && tar xzvf - . )) && ( [ -d ${HOME}/.cache/huggingface ] && rm -rf ${HOME}/.cache/huggingface || rm ${HOME}/.cache/huggingface ) && ln -s /mnt/disk${mountpoint}/huggingface ${HOME}/.cache/huggingface
			setup_hf=1
		fi

		mountpoint=$((mountpoint+1)) 
	else
		echo Disk already inuse ${inuse} ignore
	fi
done
ls -l ${HOME}/.cache
ls -l ${HOME}/.cache/huggingface
