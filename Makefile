# Simple Makefile shortcuts for running experiments
.PHONY: logistic rf knn tree adaboost bagging voting all clean

logistic:
	python -m src.experiments.train_logistic

rf:
	python -m src.experiments.train_rf

knn:
	python -m src.experiments.train_knn

tree:
	python -m src.experiments.train_tree

adaboost:
	python -m src.experiments.train_adaboost

bagging:
	python -m src.experiments.train_bagging

voting:
	python -m src.experiments.train_voting

all:
	python -m src.train

clean:
	rm -f results/*.joblib results/*_metrics.json || true
