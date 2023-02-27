#!/bin/python3

from itertools import product
import subprocess

SYSTEM = ["lorenz", "rossler", "thomas", "softrobot"]
MAP_INITIAL = ["random", "activ_f", "relax"]
PREDICTION_TYPE = ["continue", "random"]
METHOD = ["standard", "augmented"]

for comb in product(SYSTEM, MAP_INITIAL, PREDICTION_TYPE, METHOD):
    print("python", "script.py", *comb, '--debug')
    try:
        subprocess.run(["python", "script.py", *comb, '--debug'], check=True)
    except subprocess.CalledProcessError as e:
        print(e)
    print()