"""Entry point and lightweight CLI for running experiments quickly.

Usage examples:
  python main.py --model logistic   # runs the logistic experiment
  python main.py --model rf         # runs the random forest experiment
  python main.py --all              # runs all default models (same as `python -m src.train`)

This is a convenience wrapper that imports and calls experiment scripts' main() functions.
"""

import argparse
import importlib
import sys


ALLOWED_MODELS = ['logistic', 'rf', 'knn', 'tree', 'adaboost', 'bagging', 'voting']


def run_model(model_name: str):
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Allowed: {ALLOWED_MODELS}")
    module_name = f"src.experiments.train_{model_name}"
    module = importlib.import_module(module_name)
    if hasattr(module, 'main'):
        module.main()
    else:
        raise RuntimeError(f"Module {module_name} has no main() function")


def main():
    parser = argparse.ArgumentParser(description='Lightweight CLI to run experiments')
    parser.add_argument('--model', type=str, help='Name of model to run', choices=ALLOWED_MODELS)
    parser.add_argument('--all', action='store_true', help='Run all default models (python -m src.train)')
    args = parser.parse_args()

    if args.all:
        # call src.train.main()
        train_mod = importlib.import_module('src.train')
        if hasattr(train_mod, 'main'):
            train_mod.main()
            return
        else:
            print('src.train has no main()')
            sys.exit(1)

    if args.model:
        run_model(args.model)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
