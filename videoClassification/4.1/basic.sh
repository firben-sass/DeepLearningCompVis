#!/bin/sh
## ------------- specify queue name ----------------
#BSUB -q gpuv100
### ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
## ------------- specify job name ----------------
#BSUB -J testjob
## ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
## ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=20GB]"
## ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 12:00
#BSUB -o batch_output/OUTPUT_FILE%J.out
#BSUB -e batch_output/OUTPUT_FILE%J.err

source /zhome/1c/5/213743/CV/bin/activate
python -u /zhome/1c/5/213743/Vision/project_2/train_early.py