#!/bin/bash

#SBATCH --mail-user=quinlan.leishman@gmail.com   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

SYSTEM=$1
MAP_INITIAL=$2
PREDICTION_TYPE=$3
METHOD=$4
LOGDIR=$5

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

profile=slurm_${SLURM_JOB_ID}_$(hostname)

#This method for using ipyparallel with slurm was taken from https://k-d-w.org/blog/2015/05/using-ipython-for-parallel-computing-on-an-mpi-cluster-using-slurm/
echo "Creating profile ${profile}"
ipython profile create ${profile}


echo "Launching controller"
~/.local/bin/ipcontroller --ip="*" --profile=${profile} --log-to-file &
sleep 10

echo "Launching engines"
srun ~/.local/bin/ipengine --profile=${profile} --location=$(hostname) --log-to-file &
sleep 30

echo "Launching job"
python -m rescomp.opt_then_test $SYSTEM $MAP_INITIAL $PREDICTION_TYPE $METHOD $LOGDIR --parallel  --profile ${profile} --opt-ntrials 250 --test-lyap-reps 1 --ode drive_rc --algorithm population