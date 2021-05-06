# Reservoir Computing With Arbitrary Initial Conditions

This directory contains several files to help you familiarize yourself with the tools needed to complete this project. The first step, of course, is downloading the repository. In the shell, navigate to the directory where you want to store this repo and enter the command:

    git clone https://github.com/djpasseyjr/RCInitialCond
    
Now you are ready to get started!

## Dependencies
To begin, you'll need to install dependencies. Besides `numpy`, `matplotlib` and `scipy`, you need `rescomp` and `sherpa`.


    pip install rescomp
    pip install sherpa
    
## Tutorials

After cloning this repo, run `jupyter lab` and navigate to the following notebooks. Feel free to play with the examples, but don't commit your changes. You may want to do it on a separate branch.

1. `Proposal.ipynb` Gives an overview of the project.
2. `RCTrainingMethods.ipynb` Goes through using reservoir computers to learn a simple ODE. It showcases the differences between all of the different methods.
3. `LearnSherpa.ipynb` Is someone (DJ) trying to understand sherpa.

## Compute Clusters
Additionally, you will need to get both of these packages installed on the **supercomputer**. I use `conda` environments to mannage packages on the supercomputer. Learn how to make and use enviroments [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#).

There are two short scripts included in `../Scripts` for submitting jobs to the supercomputer:

    submit_script.sh
    run_in_environment.sh
    
To submit a python script to the supercomputer  enter

    bash submit_script.sh yourscript.py
    
There are some things going on on under the hood here: parameters for the `slurm` job and activating a `conda` environment. In order for this to work, you need to make a `conda` environment with exactly the name `bayesopt`, that has all the script dependencies installed.

(Alternatively, you could change the name of the enviroment in the `run_in_environment.sh` script but let's keep things simple.)





  
