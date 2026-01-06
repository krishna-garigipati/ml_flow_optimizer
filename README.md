Automated XGBoost Hyperparameter Optimization Pipeline
A production-ready MLOps pipeline that automates the tuning of an XGBoost Regressor on the California Housing dataset using Optuna for intelligent search and MLflow for experiment tracking.

🚀 Project Overview
This project demonstrates a robust machine learning workflow that moves beyond manual trial-and-error. It implements an automated system capable of finding optimal model configurations while minimizing computational waste through early stopping (pruning).

Key Features
Intelligent Search: Uses Optuna's Tree-structured Parzen Estimator (TPE) to navigate a 7-dimensional hyperparameter space.

Automated Pruning: Implements MedianPruner to terminate unpromising trials early, reducing total compute time by ~30-50%.

Experiment Tracking: Every trial is logged to MLflow, including hyperparameters, 5-fold cross-validation metrics, and trial states.

Containerized Architecture: Fully Dockerized environment to ensure 100% reproducibility across different systems.

Model Versioning: Automatically logs the best-performing model as an MLflow artifact for easy deployment.

🏗️ Technical Architecture
Model: XGBoost Regressor

Optimization: Optuna (100 trials)

Tracking: MLflow (SQLite backend)

Environment: Docker (Python 3.9-slim)

Validation: 5-Fold Cross-Validation