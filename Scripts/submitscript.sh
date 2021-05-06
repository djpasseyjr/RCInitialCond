#!/bin/sh
FILENAME=$1
OUTDIR=$2
HOURS=120
FLAGS="--ntasks=1 --cpus-per-task=1 --time=$HOURS:00:00 --chdir=$OUTDIR"
sbatch $FLAGS run_in_environment.sh $FILENAME 
