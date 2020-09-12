#!/bin/csh

#$ -M ganantha@nd.edu	 # Email address for job notification
#$ -m abe		 # Send mail when job begins, ends and aborts
#$ -q gpu		 # Specify queue
#$ -l gpu_card=1
#$ -N lr8_200e_weights         # Specify job name

#$ -pe smp 4                # Specify parallel environment and legal core size
setenv OMP_NUM_THREADS 4	         # Required modules

module load tensorflow/1.12
nvidia-smi
python DnCNN_nbn_lr8_200e_changed_lr_weights_epochs.py

