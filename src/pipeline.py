import os
import json
import time
import random
import numpy as np
import optuna
import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances

from data_loader import load_and_split_data

# ============================================================
# 1. GLOBAL REPRODUCIBILITY
# ============================================================

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# 🚨 CRITICAL: Disable any implicit MLflow behavior
mlflow.autolog(disable=True)

# ============================================================
# 2. OPTUNA OBJECTIVE (THREAD + MLflow SAFE)
# ============================================================

def objective(trial, X_train, y_train):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "random_state": SEED,
        "n_jobs": 1
    }

    # 🔥 CRITICAL FIX: clear any leaked active run
    if mlflow.active_run() is not None:
        mlflow.end_run()

    run = mlflow.start_run(run_name=f"trial_{trial.number}")

    try:
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
        mse_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr)

            preds = model.predict(X_val)
            mse_scores.append(mean_squared_error(y_val, preds))

            trial.report(np.mean(mse_scores), step=fold_idx)

            if trial.should_prune():
                mlflow.set_tag("trial_state", "PRUNED")
                raise optuna.TrialPruned()

        avg_mse = float(np.mean(mse_scores))
        avg_rmse = float(np.sqrt(avg_mse))

        mlflow.log_params(params)
        mlflow.log_metric("cv_mse", avg_mse)
        mlflow.log_metric("cv_rmse", avg_rmse)
        mlflow.log_metric("trial_number", trial.number)
        mlflow.set_tag("trial_state", "COMPLETE")

        return avg_mse

    except optuna.TrialPruned:
        raise

    except Exception:
        mlflow.set_tag("trial_state", "FAIL")
        raise

    finally:
        mlflow.end_run()


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

def run_pipeline():

    os.makedirs("outputs", exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # MLflow setup
    mlflow.set_tracking_uri("file:///app/outputs/mlruns")
    mlflow.set_experiment("optuna-xgboost-optimization")

    start_time = time.time()

    # Optuna study
    study = optuna.create_study(
        study_name="xgboost-housing-optimization",
        storage="sqlite:///optuna_study.db",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        ),
        load_if_exists=True
    )

    # Run optimization (parallel)
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=100,
        n_jobs=1
    )

    # ========================================================
    # 4. TRAIN FINAL BEST MODEL
    # ========================================================

    best_params = study.best_params
    best_params["random_state"] = SEED

    with mlflow.start_run(run_name="final_best_model"):
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(X_train, y_train)

        preds = final_model.predict(X_test)
        test_mse = mean_squared_error(y_test, preds)
        test_rmse = float(np.sqrt(test_mse))
        test_r2 = float(r2_score(y_test, preds))

        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "test_r2": test_r2
        })
        mlflow.set_tag("best_model", "true")
        mlflow.xgboost.log_model(final_model, artifact_path="model")

        # Optuna plots
        hist_path = "outputs/optimization_history.png"
        imp_path = "outputs/param_importance.png"

        plot_optimization_history(study).write_image(hist_path)
        plot_param_importances(study).write_image(imp_path)

        mlflow.log_artifact(hist_path)
        mlflow.log_artifact(imp_path)

    # ========================================================
    # 5. SAVE RESULTS.JSON
    # ========================================================

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]

    results = {
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
        "best_cv_rmse": float(np.sqrt(study.best_value)),
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "best_params": best_params,
        "optimization_time_seconds": time.time() - start_time
    }

    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Pipeline complete | RMSE: {test_rmse:.4f} | R2: {test_r2:.4f}")

# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()
