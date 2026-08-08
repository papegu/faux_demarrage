"""run_experiments.py
Simple orchestration script to run the experiments prepared in experiments_utils.py
Usage:
  python run_experiments.py --data path/to/dataset.xlsx --use-smote True

This script is intended to be run locally on a GPU-enabled machine. It will create
improve_frontiers_outputs/ with figures, CSVs and a revised manuscript copy.
"""

import argparse
import os
from experiments_utils import run_all

parser = argparse.ArgumentParser(description='Run experiments for faux_demarrage')
parser.add_argument('--data', type=str, default=None, help='Path to dataset (Excel or CSV). If omitted, the script will try default locations in the repo)')
parser.add_argument('--use-smote', action='store_true', help='Use SMOTE in training pipelines')
args = parser.parse_args()

os.makedirs('improve_frontiers_outputs', exist_ok=True)
print('Starting experiments with args:', args)
res = run_all(path=args.data, use_smote=args.use_smote)
print('Finished. Outputs:' , res)
