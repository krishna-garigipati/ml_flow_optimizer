import os
import json
import time
import random
import numpy as np
import optuna
import mlflow
import mlflow.xgboost
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances

# Import the data loader
from data_loader import load_and_split_data

# 1. Reproducibility & Seeds
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

def objective(trial, X_train, y_train):
    # 2. Search Space Definition
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "random_state": SEED,
        "n_jobs": 1  # Keep 1 to allow Optuna to manage parallel trials
    }

    with mlflow.start_run(nested=True):
        try:
            cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            mse_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr)

                preds = model.predict(X_val)
                mse = mean_squared_error(y_val, preds)
                mse_scores.append(mse)

                # Report intermediate values for pruning
                current_avg_mse = np.mean(mse_scores)
                trial.report(current_avg_mse, step=fold_idx)

                # Check if trial should be pruned
                if trial.should_prune():
                    mlflow.set_tag("trial_state", "PRUNED")
                    raise optuna.TrialPruned()

            avg_mse = np.mean(mse_scores)
            avg_rmse = np.sqrt(avg_mse)

            # Log Trial Results
            mlflow.log_params(params)
            mlflow.log_metric("cv_mse", avg_mse)
            mlflow.log_metric("cv_rmse", avg_rmse)
            mlflow.log_metric("trial_number", trial.number)
            mlflow.set_tag("trial_state", "COMPLETE")

            # Optuna will minimize this value
            return avg_mse

        except optuna.TrialPruned:
            # Re-raise so Optuna knows it was pruned
            raise
        except Exception as e:
            mlflow.set_tag("trial_state", "FAIL")
            mlflow.log_param("error", str(e))
            raise e

def run_pipeline():
    # 3. Setup
    os.makedirs("outputs", exist_ok=True)
    X_train, X_test, y_train, y_test = load_and_split_data()

    mlflow.set_tracking_uri("file:///app/outputs/mlruns")
    mlflow.set_experiment("optuna-xgboost-optimization")

    start_time = time.time()

    # 4. Study Initialization
    study = optuna.create_study(
        study_name="xgboost-housing-optimization",
        storage="sqlite:///optuna_study.db",
        direction="minimize",  # Changing to minimize for easier RMSE handling
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        ),
        load_if_exists=True
    )

    # 5. Optimization
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=100,
        n_jobs=2
    )

    # 6. Final Best Model Training
    best_params = study.best_params
    best_params["random_state"] = SEED

    with mlflow.start_run(run_name="final_best_model"):
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        test_mse = mean_squared_error(y_test, preds)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, preds)

        # Log Best Model Data
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "test_r2": test_r2
        })
        mlflow.set_tag("best_model", "true")
        mlflow.xgboost.log_model(model, artifact_path="model")

        # 7. Generate Visualizations (Requirement)
        fig_hist = plot_optimization_history(study)
        fig_imp = plot_param_importances(study)

        # Save to outputs and log as artifacts
        hist_path = "outputs/optimization_history.png"
        imp_path = "outputs/param_importance.png"
        
        # Using write_image requires 'kaleido' package
        fig_hist.write_image(hist_path)
        fig_imp.write_image(imp_path)
        
        mlflow.log_artifact(hist_path)
        mlflow.log_artifact(imp_path)

    # 8. Save results.json
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]

    results = {
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
        "best_cv_rmse": np.sqrt(study.best_value),
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "best_params": best_params,
        "optimization_time_seconds": time.time() - start_time
    }

    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Pipeline complete. Test RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

if __name__ == "__main__":
    run_pipeline()