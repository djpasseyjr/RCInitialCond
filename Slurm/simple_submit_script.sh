#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

SYSTEMS=("lorenz" "rossler" "thomas" "softrobot")
MAP_INITIALS=("random" "activ_f" "relax")
PREDICTION_TYPES=("continue" "random")
METHODS=("standard" "augmented")

for system in ${SYSTEMS[@]}; do
        for map_initial in ${MAP_INITIALS[@]}; do
                for prediction_type in ${PREDICTION_TYPES[@]}; do
                        for method in ${METHODS[@]}; do
                                FLAGS="--ntasks=1 --cpus-per-task=1 --time=3:00:00 --chdir=data"
                                sbatch $SCRIPT_DIR/opt_then_test_batch.sh $system $map_initial $prediction_type $method
                        done
                done
        done
done
