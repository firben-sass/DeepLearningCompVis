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

source /work3/s204164/work3/s204164/envs/IDLCV/bin/activate
python -u /work3/s204164/DeepLearningCompVis/projects/videoClassification/4.1/train_per_frame.py