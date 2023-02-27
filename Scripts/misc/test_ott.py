from itertools import product
import subprocess

SYSTEM = ["lorenz", "rossler", "thomas", "softrobot"]
MAP_INITIAL = ["random", "activ_f", "relax"]
PREDICTION_TYPE = ["continue", "random"]
METHOD = ["standard", "augmented"]

for sys, mi, pt, met in product(SYSTEM, MAP_INITIAL, PREDICTION_TYPE, METHOD):
    print(sys, mi, pt, met)
    try:
        subprocess.run(["python", "opt_then_test.py", sys, mi, pt, met, '--test'], check=True)
    except subprocess.CalledProcessError as e:
        print(e)
    print()