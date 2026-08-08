Branch: improve/frontiers-submission
Date: 2026-08-08
Status: Prepared experiment utilities and run script. Awaiting execution.

Actions performed so far:
- Added experiments_utils.py: utilities to compute dataset statistics, train baselines (RandomForest, XGBoost if available) and hybrid Keras model, run nested CV, and insert results into a COPY of the manuscript (additions marked in red).
- Added readme8aout.txt (this file) with trace of operations.

User choice: OK GPU — executor will attempt to use GPU if available.

Planned execution steps (to be run locally or on a GPU-enabled runner):
1) Create and activate venv (or use conda). Install requirements:
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt

2) Run the orchestrator (writes outputs to improve_frontiers_outputs/):
   python run_experiments.py --data modeles_avec_timestep/donnees_avec_faux_demarage.xlsx --use_smote False

   - The script will perform:
     * dataset summary (class counts, missing, correlation matrix)
     * nested cross-validation (5 folds) training RandomForest, XGBoost (if installed) and Hybrid Keras
     * save fold-level probabilities and metrics to improve_frontiers_outputs/
     * generate figures in improve_frontiers_outputs/figures
     * create a copy of the manuscript with added sections (article_..._rev.docx) with additions marked in red

3) After the run, inspect improve_frontiers_outputs/ and commit outputs if desired.

Notes and resource guidance:
- The GPU option is selected; ensure CUDA/cuDNN and a tensorflow-gpu compatible wheel are installed. requirements.txt contains a generic tensorflow specification — please install a GPU-enabled build appropriate for your platform (e.g., tensorflow==2.12.0 or tensorflow-gpu where applicable).
- First pass uses bootstrap=200 for speed; adjust in experiments_utils.py if you want bootstrap=1000.

Next steps I will do for you in the repository:
- Add run_experiments.py and requirements.txt (done in branch).
- Update manuscript with automated results once runs are executed locally or in a runner with GPU. I cannot execute heavy training inside this environment — I prepared all scripts for you and will commit outputs into the branch if you run them and push results back, or I can continue if you provide access to a runner.

Trace of commits in this branch (high level):
- add readme8aout.txt
- Add experiments_utils.py
- Add run_experiments.py and requirements.txt

If you want me to proceed with running experiments on a remote CI/runner, provide credentials or instructions; otherwise run the commands above locally and push the outputs; I will then update the manuscript and make final commits.
