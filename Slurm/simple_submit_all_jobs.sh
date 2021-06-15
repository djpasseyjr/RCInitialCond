#!/bin/bash

RCDIR="$(dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) )"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

submit_job () {
        job=$1
        run_time=$2
        echo $job
        TIMESTAMP=$(date +"%m-%d-%y:%H:%M:%S")
        LOGDIR="$RCDIR"/Data/"$job"_"$TIMESTAMP"
        LOGDIR="$HOME/fsl_groups/fslg_chaos/$job\_$TIMESTAMP"
        mkdir -p $LOGDIR
        echo $LOGDIR
        job_args=$(sed 's/_/ /g' <<<"$job")
        job_args=$(sed 's/activ f/activ_f/g' <<<"$job_args")
        echo $job_args
        FLAGS="-o $LOGDIR/slurm-%a.out --time=$run_time --ntasks=1 --nodes=1 --mem-per-cpu=1024M -J $job"
        sbatch $FLAGS $SCRIPT_DIR/opt_then_test_batch.sh $job_args $RCDIR $LOGDIR
}

SYSTEMS=("lorenz" "rossler" "thomas" "softrobot")
MAP_INITIALS=("random" "activ_f" "relax")
PREDICTION_TYPES=("continue" "random")
METHODS=("standard" "augmented")

declare -a JOBS=("thomas_activ_f_random_standard"
        "thomas_activ_f_random_augmented"
        "thomas_relax_continue_standard"
        "thomas_relax_random_standard"
        "thomas_relax_random_augmented"
        "lorenz_relax_continue_augmented"
        "rossler_activ_f_continue_augmented"
        "rossler_relax_continue_augmented"
        "rossler_relax_random_augmented"
        "thomas_random_continue_standard"
        "thomas_random_continue_augmented"
        "thomas_random_random_standard"
        "thomas_random_random_augmented"
        "thomas_activ_f_continue_augmented")

for job in ${JOBS[@]}; do
        submit_job $job "24:00:00"
done

declare -a JOBS=(
        "lorenz_random_continue_standard"
        "lorenz_random_continue_augmented"
        "lorenz_random_random_standard"
        "lorenz_random_random_augmented"
        "lorenz_activ_f_continue_standard"
        "lorenz_activ_f_continue_augmented"
        "lorenz_activ_f_random_standard"
        "lorenz_activ_f_random_augmented"
        "lorenz_relax_continue_standard"
        "lorenz_relax_continue_augmented"
        "lorenz_relax_random_standard"
        "lorenz_relax_random_augmented"
        "rossler_random_continue_standard"
        "rossler_random_continue_augmented"
        "rossler_random_random_standard"
        "rossler_random_random_augmented"
        "rossler_activ_f_continue_standard"
        "rossler_activ_f_continue_augmented"
        "rossler_activ_f_random_standard"
        "rossler_activ_f_random_augmented"
        "rossler_relax_continue_standard"
        "rossler_relax_continue_augmented"
        "rossler_relax_random_standard"
        "rossler_relax_random_augmented"
        "thomas_random_continue_standard"
        "thomas_random_continue_augmented"
        "thomas_random_random_standard"
        "thomas_random_random_augmented"
        "thomas_activ_f_continue_standard"
        "thomas_activ_f_continue_augmented"
        "thomas_activ_f_random_standard"
        "thomas_activ_f_random_augmented"
        "thomas_relax_continue_standard"
        "thomas_relax_continue_augmented"
        "thomas_relax_random_standard"
        "thomas_relax_random_augmented"
        "softrobot_random_continue_standard"
        "softrobot_random_continue_augmented"
        "softrobot_random_random_standard"
        "softrobot_random_random_augmented"
        "softrobot_activ_f_continue_standard"
        "softrobot_activ_f_continue_augmented"
        "softrobot_activ_f_random_standard"
        "softrobot_activ_f_random_augmented"
        "softrobot_relax_continue_standard"
        "softrobot_relax_continue_augmented"
        "softrobot_relax_random_standard"
        "softrobot_relax_random_augmented"
)

for job in ${JOBS[@]}; do
        submit_job $job "12:00:00"
done