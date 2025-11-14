#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J segmentation_job
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 5GB of memory per core/slot -- 
#BSUB -R "rusage[mem=5GB]"
### -- request one GPU --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 12:00
### -- set the email address -- 
#BSUB -u s204164@dtu.dk
### -- send notification at completion -- 
#BSUB -N
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o batch_output/Output_%J.out
#BSUB -e batch_output/Error_%J.err

# here follow the commands you want to execute with input.in as the input file
nvidia-smi
module load cuda/11.6
source /work3/s204164/work3/s204164/etc/profile.d/conda.sh
conda activate IDLCV
python -u /work3/s204164/DeepLearningCompVis/segmentation/train.py \
	--run-name segmentation_DRIVE \
	--epochs 500 \
	--dataset DRIVE