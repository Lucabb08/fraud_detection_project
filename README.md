Project_organization
├── README.md           # Setup and usage instructions
├── PROPOSAL.md         # Project proposal
├── environment.yml     # Conda dependencies
├── requirements.txt    # Pip dependencies
├── main.py             # Entry point
├── src/                # Source code modules
│   ├── __init__.py
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── models.py       # Model definitions
│   └── evaluation.py   # Evaluation and visualization
├── data/
│   └── raw/            # Original dataset
├── results/            
└── notebooks/          

# Setup & Usage

This repository has been structured according to the project template. Fill `environment.yml` or `requirements.txt` with the project dependencies, place raw data in `data/raw/`, implement data loading in `src/data_loader.py`, model code in `src/models.py`, and evaluation/visualization in `src/evaluation.py`.

---

## Quick start (how to run experiments)

1) Download the dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it at `data/raw/creditcard.csv`.


2) Train one model (example: logistic regression):

   - python -m src.experiments.train_logistic

   This will train the pipeline, print precision/recall/F1/AUPRC and save the pipeline to `results/logistic_pipeline.joblib`.

3) Train all default models:

   - python -m src.train
   - or use a shortcut with Make (from the repo root): `make all`

   This will train the default builders, print metrics, and save each model into `results/`.

---

## Shortcuts with Make (optional)

Instead of typing the full python command you can use Makefile shortcuts (run from the project root):

- `make logistic`   → runs only logistic regression experiment
- `make rf`         → runs Random Forest experiment
- `make knn`        → runs KNN experiment
- `make tree`       → runs Decision Tree experiment
- `make adaboost`   → runs AdaBoost experiment
- `make bagging`    → runs Bagging experiment
- `make voting`     → runs Voting ensemble experiment
- `make all`        → trains all default models (same as `python -m src.train`)

These shortcuts are convenience aliases to run experiments quickly from the terminal.

