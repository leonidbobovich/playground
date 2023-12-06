#!/bin/bash

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
		sudo mkdir /mnt/disk${mountpoint}
		sudo mount /dev/${d} /mnt/disk${mountpoint}
		sudo chown -R ${USER} /mnt/disk${mountpoint}
		echo Disk ${d} mounted on  /mnt/disk${mountpoint}
		if [ "${setup_hf}" == "0" ]; then	
			cache_hf_dir=${HOME}/.cache/huggingface
			echo Setting up hf on new disk ${d} /mnt/disk${mountpoint}/huggingface
			[ ! -d /mnt/disk${mountpoint}/huggingface ] && mkdir /mnt/disk${mountpoint}/huggingface/
			[ -d ${HOME}/.cache/huggingface ] && mv ${HOME}/.cache/huggingface/* /mnt/disk${mountpoint}/huggingface/
			[[ -L "${HOME}/.cache/huggingface" ]] &&  rm "${HOME}/.cache/huggingface" || rm -rf "${HOME}/.cache/huggingface"
			ln -s /mnt/disk${mountpoint}/huggingface ${HOME}/.cache/huggingface
			setup_hf=1
		fi

		mountpoint=$((mountpoint+1)) 
	else
		echo Disk already inuse ${inuse} ignore
	fi
done
