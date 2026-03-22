"""
train_model.py
---------------
Reads preprocessed churn data, performs hyperparameter tuning
with GridSearchCV + StratifiedKFold cross-validation, trains
6 classification models, compares them, and saves the best
model pipeline as models/churn_model.pkl.

Primary metric  : ROC-AUC
Secondary metric: Accuracy

Models trained:
  1. Logistic Regression
  2. Random Forest
  3. Gradient Boosting
  4. XGBoost
  5. K-Nearest Neighbors
  6. Support Vector Machine

Input  : data/preprocessed/churn_preprocessed.csv
         data/preprocessed/feature_columns.json
Output : models/churn_model.pkl
         models/model_comparison.csv
         models/best_model_info.json
"""

import os
import json
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent.parent
PREPROC_PATH    = BASE_DIR / "data" / "preprocessed" / "churn_preprocessed.csv"
FEAT_COLS_PATH  = BASE_DIR / "data" / "preprocessed" / "feature_columns.json"
MODELS_DIR      = BASE_DIR / "models"
MODEL_PATH      = MODELS_DIR / "churn_model.pkl"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
BEST_INFO_PATH  = MODELS_DIR / "best_model_info.json"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════

def load_data():
    log.info(f"Loading: {PREPROC_PATH}")
    df = pd.read_csv(PREPROC_PATH)

    with open(FEAT_COLS_PATH) as f:
        metadata = json.load(f)

    target       = metadata['target_column']
    feature_cols = [c for c in metadata['feature_columns'] if c in df.columns]

    X = df[feature_cols]
    y = df[target]

    log.info(f"Shape: {X.shape} | Churn rate: {y.mean():.2%}")
    return X, y, feature_cols


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — MODEL DEFINITIONS + HYPERPARAMETER GRIDS
# ═════════════════════════════════════════════════════════════════════════════

def get_models_and_grids():
    """
    Each entry is a tuple of (pipeline, param_grid).
    Param keys follow sklearn convention: 'stepname__paramname'.
    Grids are kept focused — wide grids on 6 models would take hours.
    """

    models = {

        "Logistic Regression": (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                ))
            ]),
            {
                'model__C'      : [0.01, 0.1, 1, 10],
                'model__solver' : ['lbfgs', 'liblinear'],
                'model__penalty': ['l2'],
            }
        ),

        "Random Forest": (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            {
                'model__n_estimators'     : [100, 200],
                'model__max_depth'        : [4, 6, None],
                'model__min_samples_split': [2, 5],
            }
        ),

        "Gradient Boosting": (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingClassifier(
                    random_state=42
                ))
            ]),
            {
                'model__n_estimators' : [100, 200],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth'    : [3, 4],
            }
        ),

        "XGBoost": (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', XGBClassifier(
                    scale_pos_weight=3,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    verbosity=0
                ))
            ]),
            {
                'model__n_estimators' : [100, 200],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth'    : [3, 4],
                'model__subsample'    : [0.8, 1.0],
            }
        ),

        "KNN": (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', KNeighborsClassifier(n_jobs=-1))
            ]),
            {
                'model__n_neighbors': [5, 7, 11],
                'model__weights'    : ['uniform', 'distance'],
                'model__metric'     : ['euclidean', 'manhattan'],
            }
        ),

        "LightGBM": (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', LGBMClassifier(
                class_weight='balanced',
                random_state=42,
                verbosity=-1
            ))
             ]),
        {
            'model__n_estimators' : [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth'    : [4, 6],
            'model__num_leaves'   : [31, 50],
        }
        ),
        "SVM": (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(
                    class_weight='balanced',
                    probability=True,
                    random_state=42
                ))
            ]),
            {
                'model__C'     : [0.1, 1, 10],
                'model__kernel': ['rbf', 'linear'],
                'model__gamma' : ['scale', 'auto'],
            }
        ),
    }

    return models


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — HYPERPARAMETER TUNING + CROSS VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def tune_and_evaluate(X_train, X_test, y_train, y_test):
    """
    For each model:
      1. GridSearchCV with 5-fold StratifiedKFold to find best params
      2. Refit best estimator on full training set
      3. Evaluate on held-out test set
      4. Run 5-fold CV on best estimator for final ROC-AUC estimate
    """
    models_and_grids = get_models_and_grids()
    cv               = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results          = []
    best_estimators  = {}

    for name, (pipeline, param_grid) in models_and_grids.items():
        log.info(f"{'─'*55}")
        log.info(f"Tuning: {name}")

        # ── GridSearchCV ──────────────────────────────────────────
        grid_search = GridSearchCV(
            estimator  = pipeline,
            param_grid = param_grid,
            cv         = cv,
            scoring    = 'roc_auc',   # primary metric
            n_jobs     = -1,
            verbose    = 0,
            refit      = True         # refit best model on full train set
        )
        grid_search.fit(X_train, y_train)

        best_pipeline  = grid_search.best_estimator_
        best_params    = grid_search.best_params_
        best_cv_score  = grid_search.best_score_

        log.info(f"  Best params : {best_params}")
        log.info(f"  CV ROC-AUC  : {best_cv_score:.4f}")

        # ── Test set evaluation ───────────────────────────────────
        y_pred      = best_pipeline.predict(X_test)
        y_pred_prob = best_pipeline.predict_proba(X_test)[:, 1]

        roc_auc   = roc_auc_score(y_test, y_pred_prob)
        accuracy  = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)

        # ── Final cross-val on best estimator ─────────────────────
        final_cv = cross_val_score(
            best_pipeline, X_train, y_train,
            cv=cv, scoring='roc_auc', n_jobs=-1
        )

        log.info(
            f"  Test ROC-AUC: {roc_auc:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"F1: {f1:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f}"
        )

        results.append({
            'Model'           : name,
            'ROC-AUC'         : round(roc_auc,        4),
            'Accuracy'        : round(accuracy,        4),
            'F1 Score'        : round(f1,              4),
            'Precision'       : round(precision,       4),
            'Recall'          : round(recall,          4),
            'CV ROC-AUC Mean' : round(final_cv.mean(), 4),
            'CV ROC-AUC Std'  : round(final_cv.std(),  4),
            'Best Params'     : str(best_params),
        })

        best_estimators[name] = best_pipeline

    results_df = pd.DataFrame(results).sort_values(
        by=['ROC-AUC', 'Accuracy'], ascending=False
    ).reset_index(drop=True)

    return results_df, best_estimators


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — DETAILED REPORT FOR BEST MODEL
# ═════════════════════════════════════════════════════════════════════════════

def detailed_report(pipeline, X_test, y_test, model_name):
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    log.info(f"\n{'='*55}")
    log.info(f"BEST MODEL  : {model_name}")
    log.info(f"ROC-AUC     : {roc_auc_score(y_test, y_pred_prob):.4f}")
    log.info(f"Accuracy    : {accuracy_score(y_test, y_pred):.4f}")
    log.info(f"\nClassification Report:\n"
             f"{classification_report(y_test, y_pred, target_names=['Not Churned','Churned'])}")
    log.info(f"{'='*55}\n")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════

def save_outputs(results_df, best_estimators, X_test, y_test, feature_cols):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Model comparison CSV ──────────────────────────────────────
    results_df.to_csv(COMPARISON_PATH, index=False)
    log.info(f"Saved comparison → {COMPARISON_PATH}")

    # ── Best model ────────────────────────────────────────────────
    best_name     = results_df.iloc[0]['Model']
    best_pipeline = best_estimators[best_name]

    joblib.dump(best_pipeline, MODEL_PATH)
    log.info(f"Saved best model ({best_name}) → {MODEL_PATH}")

    # ── Best model info JSON ──────────────────────────────────────
    best_row = results_df.iloc[0].to_dict()
    info = {
        "best_model"     : best_name,
        "roc_auc"        : best_row['ROC-AUC'],
        "accuracy"       : best_row['Accuracy'],
        "f1_score"       : best_row['F1 Score'],
        "precision"      : best_row['Precision'],
        "recall"         : best_row['Recall'],
        "cv_roc_auc_mean": best_row['CV ROC-AUC Mean'],
        "cv_roc_auc_std" : best_row['CV ROC-AUC Std'],
        "best_params"    : best_row['Best Params'],
        "feature_columns": feature_cols,
        "model_path"     : str(MODEL_PATH),
    }
    with open(BEST_INFO_PATH, 'w') as f:
        json.dump(info, f, indent=2)
    log.info(f"Saved model info → {BEST_INFO_PATH}")

    # ── Detailed report for best model ────────────────────────────
    detailed_report(best_pipeline, X_test, y_test, best_name)

    return best_name


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_training():
    log.info("=" * 55)
    log.info("START: Model Training Pipeline")
    log.info("=" * 55)

    # 1. Load
    X, y, feature_cols = load_data()

    # 2. Train/test split — stratified to preserve churn ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    log.info(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    log.info(f"Train churn rate: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    # 3. Tune and evaluate all models
    results_df, best_estimators = tune_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # 4. Print comparison table
    log.info("\nModel Comparison (sorted by ROC-AUC):")
    log.info("\n" + results_df[
        ['Model', 'ROC-AUC', 'Accuracy', 'F1 Score',
         'CV ROC-AUC Mean', 'CV ROC-AUC Std']
    ].to_string(index=False))

    # 5. Save everything
    best_name = save_outputs(
        results_df, best_estimators, X_test, y_test, feature_cols
    )

    log.info("=" * 55)
    log.info(f"DONE: Best model = {best_name}")
    log.info("=" * 55)


if __name__ == "__main__":
    run_training()