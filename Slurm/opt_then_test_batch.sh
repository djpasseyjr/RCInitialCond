#!/bin/bash
#***** NOTE: run this using: sg fslg_chaos "sbatch thefilename"

#SBATCH --time=04:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
#SBATCH -J "opt_then_test"   # job name
#SBATCH --mail-user=taylorpool.27@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

SYSTEM=$1
MAP_INITIAL=$2
PREDICTION_TYPE=$3
METHOD=$4
LOGDIR=$5

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source $RCDIR/venv/bin/activate
cd $RCDIR/Scripts
python opt_then_test.py $SYSTEM $MAP_INITIAL $PREDICTION_TYPE $METHOD $LOGDIR