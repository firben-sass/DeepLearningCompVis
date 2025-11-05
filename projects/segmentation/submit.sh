#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J segmentation_job
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 00:15 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
##BSUB -B 
### -- send notification at completion -- 
##BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o outputs/Output_%J.out 
#BSUB -e outputs/Error_%J.err 

# here follow the commands you want to execute with input.in as the input file
source /work3/s204164/work3/s204164/etc/profile.d/conda.sh
conda activate IDLCV
python -u /work3/s204164/IDLCV/segmentation_exercise/train.py