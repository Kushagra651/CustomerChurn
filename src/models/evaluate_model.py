"""
evaluate_model.py
------------------
Loads the best saved model and model comparison CSV,
generates evaluation plots and a full metrics report.

Outputs saved to models/evaluation/:
  - model_comparison.png     : bar chart comparing all models
  - precision_recall.png     : PR curve for best model
  - evaluation_report.json   : full metrics summary

Input  : models/churn_model.pkl
         models/model_comparison.csv
         models/best_model_info.json
         data/preprocessed/churn_preprocessed.csv
         data/preprocessed/feature_columns.json
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent.parent
MODELS_DIR      = BASE_DIR / "models"
EVAL_DIR        = MODELS_DIR / "evaluation"
MODEL_PATH      = MODELS_DIR / "churn_model.pkl"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
BEST_INFO_PATH  = MODELS_DIR / "best_model_info.json"
PREPROC_PATH    = BASE_DIR / "data" / "preprocessed" / "churn_preprocessed.csv"
FEAT_COLS_PATH  = BASE_DIR / "data" / "preprocessed" / "feature_columns.json"

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'      : 150,
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'font.family'     : 'sans-serif',
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 11,
})

COLORS = {
    'primary'  : '#4F46E5',
    'secondary': '#10B981',
    'accent'   : '#F59E0B',
    'danger'   : '#EF4444',
    'gray'     : '#6B7280',
    'light'    : '#E5E7EB',
}


# ═════════════════════════════════════════════════════════════════════════════
# LOAD DATA + MODEL
# ═════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(PREPROC_PATH)

    with open(FEAT_COLS_PATH) as f:
        metadata = json.load(f)

    target       = metadata['target_column']
    feature_cols = [c for c in metadata['feature_columns'] if c in df.columns]

    X = df[feature_cols]
    y = df[target]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Test set: {X_test.shape[0]} rows")
    return X_test, y_test


def load_model():
    pipeline = joblib.load(MODEL_PATH)
    log.info(f"Loaded model from {MODEL_PATH}")

    with open(BEST_INFO_PATH) as f:
        info = json.load(f)

    log.info(f"Best model: {info['best_model']}")
    return pipeline, info


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — MODEL COMPARISON BAR CHART
# ═════════════════════════════════════════════════════════════════════════════

def plot_model_comparison():
    df = pd.read_csv(COMPARISON_PATH)

    metrics = ['ROC-AUC', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
    # Keep only metrics that exist in the CSV
    metrics = [m for m in metrics if m in df.columns]

    n_models  = len(df)
    n_metrics = len(metrics)
    x         = np.arange(n_models)
    width     = 0.15
    bar_colors= [COLORS['primary'], COLORS['secondary'],
                 COLORS['accent'],  COLORS['danger'], COLORS['gray']]

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (metric, color) in enumerate(zip(metrics, bar_colors)):
        offset = (i - n_metrics / 2) * width + width / 2
        bars   = ax.bar(
            x + offset, df[metric], width,
            label=metric, color=color, alpha=0.85,
            edgecolor='white', linewidth=0.5
        )
        # Value labels on top of bars
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=7, color='#374151'
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — All Metrics', fontweight='bold', pad=15)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
    ax.axhline(y=0.5, color=COLORS['light'], linestyle='--', linewidth=0.8)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    path = EVAL_DIR / "model_comparison.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    log.info(f"Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — PRECISION-RECALL CURVE
# ═════════════════════════════════════════════════════════════════════════════

def plot_precision_recall(pipeline, X_test, y_test, model_name):
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    # Find threshold closest to best F1
    f1_scores  = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx   = np.argmax(f1_scores)
    best_thresh= thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_prec  = precision[best_idx]
    best_rec   = recall[best_idx]
    best_f1    = f1_scores[best_idx]

    # Baseline (random classifier = churn rate)
    baseline = y_test.mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: PR Curve ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(recall, precision, color=COLORS['primary'],
            linewidth=2.5, label=f'PR Curve (AP = {avg_precision:.3f})')
    ax.axhline(y=baseline, color=COLORS['danger'], linestyle='--',
               linewidth=1.2, label=f'Baseline (churn rate = {baseline:.2f})')
    ax.scatter(best_rec, best_prec, color=COLORS['accent'],
               s=120, zorder=5, label=f'Best F1 = {best_f1:.3f}')
    ax.annotate(
        f'  Threshold={best_thresh:.2f}\n  F1={best_f1:.3f}',
        xy=(best_rec, best_prec),
        fontsize=8, color='#374151'
    )
    ax.fill_between(recall, precision, alpha=0.08, color=COLORS['primary'])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve\n{model_name}', fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9)
    ax.set_facecolor('#FAFAFA')

    # ── Right: Precision & Recall vs Threshold ────────────────────
    ax2 = axes[1]
    ax2.plot(thresholds, precision[:-1], color=COLORS['primary'],
             linewidth=2, label='Precision')
    ax2.plot(thresholds, recall[:-1], color=COLORS['secondary'],
             linewidth=2, label='Recall')
    ax2.plot(thresholds, f1_scores[:-1], color=COLORS['accent'],
             linewidth=2, linestyle='--', label='F1 Score')
    ax2.axvline(x=best_thresh, color=COLORS['danger'], linestyle=':',
                linewidth=1.5, label=f'Best threshold = {best_thresh:.2f}')
    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Precision / Recall / F1 vs Threshold', fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])
    ax2.legend(fontsize=9)
    ax2.set_facecolor('#FAFAFA')

    fig.patch.set_facecolor('white')
    plt.tight_layout()

    path = EVAL_DIR / "precision_recall.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    log.info(f"Saved → {path}")

    return avg_precision, best_thresh, best_f1


# ═════════════════════════════════════════════════════════════════════════════
# SAVE EVALUATION REPORT
# ═════════════════════════════════════════════════════════════════════════════

def save_evaluation_report(pipeline, X_test, y_test, info,
                            avg_precision, best_thresh, best_f1):
    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)[:, 1]
    cm          = confusion_matrix(y_test, y_pred)

    report = {
        "best_model"        : info['best_model'],
        "test_set_size"     : int(len(y_test)),
        "churn_rate"        : float(round(y_test.mean(), 4)),
        "roc_auc"           : float(round(roc_auc_score(y_test, y_prob), 4)),
        "accuracy"          : float(round(accuracy_score(y_test, y_pred), 4)),
        "f1_score"          : float(round(f1_score(y_test, y_pred), 4)),
        "precision"         : float(round(precision_score(y_test, y_pred), 4)),
        "recall"            : float(round(recall_score(y_test, y_pred), 4)),
        "avg_precision"     : float(round(avg_precision, 4)),
        "best_threshold"    : float(round(best_thresh, 4)),
        "best_f1_threshold" : float(round(best_f1, 4)),
        "cv_roc_auc_mean"   : info.get('cv_roc_auc_mean'),
        "cv_roc_auc_std"    : info.get('cv_roc_auc_std'),
        "confusion_matrix"  : {
            "true_negative" : int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive" : int(cm[1][1]),
        },
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=['Not Churned', 'Churned'],
            output_dict=True
        )
    }

    path = EVAL_DIR / "evaluation_report.json"
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    log.info(f"Saved → {path}")

    # Print summary
    log.info(f"\n{'='*55}")
    log.info(f"EVALUATION SUMMARY — {info['best_model']}")
    log.info(f"{'='*55}")
    log.info(f"  ROC-AUC   : {report['roc_auc']}")
    log.info(f"  Accuracy  : {report['accuracy']}")
    log.info(f"  F1 Score  : {report['f1_score']}")
    log.info(f"  Precision : {report['precision']}")
    log.info(f"  Recall    : {report['recall']}")
    log.info(f"  Avg Prec  : {report['avg_precision']}")
    log.info(f"  Best Thresh: {report['best_threshold']}")
    log.info(f"\n  Confusion Matrix:")
    log.info(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    log.info(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    log.info(f"{'='*55}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation():
    log.info("=" * 55)
    log.info("START: Model Evaluation")
    log.info("=" * 55)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    X_test, y_test   = load_data()
    pipeline, info   = load_model()

    # Plot 1 — model comparison
    log.info("Generating model comparison chart...")
    plot_model_comparison()

    # Plot 2 — precision recall curve
    log.info("Generating precision-recall curve...")
    avg_precision, best_thresh, best_f1 = plot_precision_recall(
        pipeline, X_test, y_test, info['best_model']
    )

    # Save full report
    save_evaluation_report(
        pipeline, X_test, y_test, info,
        avg_precision, best_thresh, best_f1
    )

    log.info("=" * 55)
    log.info("DONE: Evaluation Complete")
    log.info(f"Outputs saved to {EVAL_DIR}")
    log.info("=" * 55)


if __name__ == "__main__":
    run_evaluation()