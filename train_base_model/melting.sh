#!/bin/bash -l
# Use the current working directory
#SBATCH -D ./
# Use the current environment for this job
#SBATCH --export=ALL
#SBATCH --export XDG_RUNTIME_DIR=/users/fourteen/ying/1/

# Define job name
#SBATCH -J TFGPU
# Define a standard output/error file. When the job is running, %N will be replaced by the name of 
# the first node where the job runs, %j will be replaced by job id number.
#SBATCH -o tensorflow.%N.%j.out
# Set time limit in format a-bb:cc:dd, where a is days, b is hours, c is minutes, and d is seconds.
#SBATCH -t 1-00:00:00
# Request the memory on the node or request memory per core
# PLEASE don't set the memory option as we should use the default memory which is based on the number of cores 
##SBATCH --mem=90GB or #SBATCH --mem-per-cpu=9000M
# Insert your own username to get e-mail notifications (note: keep just one "#" before SBATCH)
##SBATCH --fourteen=Ying.jiang@liverpool.ac.uk
# Notify user by email when certain event types occur
#SBATCH --mail-type=ALL
#
# Set your maximum stack size to unlimited
ulimit -s unlimited
# Set OpenMP thread number
export OMP_NUM_THREADS=$SLURM_NTASKS

# Load tensorflow and relevant modules
module load apps/anaconda3/5.2.0-python37

#use source activate gpu to get the gpu virtual environment
source activate py39



echo =========================================================   
echo SLURM job: submitted  date = `date`      
date_start=`date +%s`

hostname

echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "GPU_DEVICE_ORDINAL   : $GPU_DEVICE_ORDINAL"

echo "Running GPU jobs:"
conda env list
echo "Running GPU job train.py:"
python train.py


#deactivate the gpu virtual environment
source deactivate py39

date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================   
echo SLURM job: finished   date = `date`   
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================   
