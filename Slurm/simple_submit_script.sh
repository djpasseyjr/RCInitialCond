#!/bin/bash
RCDIR="$(dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) )"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

SYSTEMS=("lorenz" "rossler" "thomas" "softrobot")
MAP_INITIALS=("random" "activ_f" "relax")
PREDICTION_TYPES=("continue" "random")
METHODS=("standard" "augmented")

SYSTEMS=("softrobot")

for system in ${SYSTEMS[@]}; do
        for map_initial in ${MAP_INITIALS[@]}; do
                for prediction_type in ${PREDICTION_TYPES[@]}; do
                        for method in ${METHODS[@]}; do
                                TIMESTAMP=$(date +"%m-%d-%y:%H:%M:%S")
                                LOGDIR=$RCDIR"/Data/"$system"_"$map_initial"_"$prediction_type"_"$method"_"$TIMESTAMP
                                mkdir $LOGDIR
                                FLAGS="-o $LOGDIR/slurm-%a.out --time=04:00:00 --ntasks=1 --nodes=1 --mem-per-cpu=1024M -J "$system"_"$map_initial"_"$prediction_type"_"$method
                                sbatch $FLAGS $SCRIPT_DIR/opt_then_test_batch.sh $system $map_initial $prediction_type $method $RCDIR $LOGDIR
                        done
                done
        done
done
