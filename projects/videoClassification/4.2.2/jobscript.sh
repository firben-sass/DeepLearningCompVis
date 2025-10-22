!/bin/sh
### General options
### –- specify queue --
#BSUB -q c02516
### -- set the job Name --
#BSUB -J DualStreamNetwork_job
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request ?GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s244405@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o simple_script_job.out
#BSUB -e simple_script_job.err
# -- end of LSF options --

# Activate python virtual
source /zhome/dc/a/215453/Desktop/DeepLearningCompVis/env/bin/activate
# Run script
python3 /zhome/dc/a/215453/Desktop/DeepLearningCompVis/exercises/Exercise_2_1_H_PC/IDLCV_Exercise_2_1_H_PC.pys