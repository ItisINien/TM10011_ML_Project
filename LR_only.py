# %%
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    shap = None

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer, roc_auc_score, roc_curve
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from worcliver.load_data import load_data


RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
N_RANDOM_SEARCH_ITERATIONS = 20
N_JOBS = 1
F_BETA = 2
ROC_OUTPUT_PATH = "roc_curve_nested_cv_lr_only.png"
SHAP_OUTPUT_DIR = "shap_outputs_nested_cv_lr_only"


class DataFrameRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler_ = RobustScaler()

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.feature_names_in_)
        return pd.DataFrame(X)

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = X_df.columns.to_list()
        self.scaler_.fit(X_df, y)
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X)
        transformed = self.scaler_.transform(X_df)
        return pd.DataFrame(transformed, columns=self.feature_names_in_, index=X_df.index)


def build_pipeline():
    final_lr = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            ("scaler", DataFrameRobustScaler()),
            ("clf", final_lr),
        ]
    )


def compute_fbeta_from_proba(y_true, y_proba, beta=F_BETA, threshold=0.5):
    y_pred = (np.array(y_proba) >= threshold).astype(int)
    return float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))


def summarize_params(best_params_per_fold):
    summary = {}
    for key in best_params_per_fold[0]:
        values = [params[key] for params in best_params_per_fold]
        summary[key] = Counter(values).most_common(1)[0][0]
    return summary


def save_roc_curve(y_true, y_scores, roc_auc, output_path=ROC_OUTPUT_PATH):
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
        linewidth=2,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Nested CV ROC Curve (LR no feature selection)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve to: {Path(output_path).resolve()}")


def run_nested_shap_summary(nested_shap_payload, output_dir=SHAP_OUTPUT_DIR):
    if shap is None:
        print("\nNested SHAP summary skipped: package 'shap' is not installed.")
        return

    if not nested_shap_payload:
        print("\nNested SHAP summary skipped: no SHAP payload available.")
        return

    all_features = nested_shap_payload[0]["feature_names"]
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    combined_shap = pd.concat(
        [
            pd.DataFrame(
                fold_payload["shap_values"],
                columns=fold_payload["feature_names"],
                index=fold_payload["index"],
            )
            for fold_payload in nested_shap_payload
        ],
        axis=0,
    ).sort_index()

    importance_df = (
        pd.DataFrame(
            {
                "feature": all_features,
                "mean_abs_shap": np.abs(combined_shap.to_numpy()).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    importance_df.to_csv(output_path / "nested_shap_importance.csv", index=False)

    print("\n" + "=" * 40)
    print("Nested CV SHAP mean absolute importance")
    print(importance_df.head(20).to_string(index=False))
    print(f"Saved nested SHAP summary to: {output_path.resolve()}")


def run_nested_cv(X_trainval, y_trainval):
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    param_dist = {
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
    }

    all_y_outer = []
    all_y_outer_proba = []
    outer_fold_aucs = []
    outer_fold_f2s = []
    best_params_per_fold = []
    nested_shap_payload = []

    print("Start nested cross-validation on the 80% train/validation set...")

    for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(
        outer_cv.split(X_trainval, y_trainval),
        start=1,
    ):
        print(f"\n--- Outer fold {fold_idx} ---")

        X_outer_train = X_trainval.iloc[outer_train_idx].copy()
        X_outer_val = X_trainval.iloc[outer_val_idx].copy()
        y_outer_train = y_trainval.iloc[outer_train_idx]
        y_outer_val = y_trainval.iloc[outer_val_idx]

        search = RandomizedSearchCV(
            estimator=build_pipeline(),
            param_distributions=param_dist,
            n_iter=min(N_RANDOM_SEARCH_ITERATIONS, len(param_dist["clf__C"])),
            cv=inner_cv,
            scoring=make_scorer(fbeta_score, beta=F_BETA, zero_division=0),
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
        )
        search.fit(X_outer_train, y_outer_train)

        best_model = search.best_estimator_

        y_outer_proba = best_model.predict_proba(X_outer_val)[:, 1]
        fold_auc = roc_auc_score(y_outer_val, y_outer_proba)
        fold_f2 = compute_fbeta_from_proba(y_outer_val, y_outer_proba)

        all_y_outer.extend(y_outer_val)
        all_y_outer_proba.extend(y_outer_proba)
        outer_fold_aucs.append(fold_auc)
        outer_fold_f2s.append(fold_f2)
        best_params_per_fold.append(search.best_params_)

        if shap is not None:
            scaled_train = pd.DataFrame(
                best_model.named_steps["scaler"].transform(X_outer_train),
                columns=X_outer_train.columns,
                index=X_outer_train.index,
            )
            background_size = min(100, len(scaled_train))
            background = scaled_train.sample(background_size, random_state=RANDOM_STATE)
            explainer = shap.LinearExplainer(best_model.named_steps["clf"], background)
            shap_values = explainer(scaled_train)
            nested_shap_payload.append(
                {
                    "feature_names": scaled_train.columns.to_list(),
                    "shap_values": shap_values.values,
                    "index": scaled_train.index,
                }
            )

        print(f"Best params: {search.best_params_}")
        print(f"Features used: {X_outer_train.shape[1]}")
        print(f"Outer fold ROC-AUC: {fold_auc:.3f}")
        print(f"Outer fold F{F_BETA}-score: {fold_f2:.3f}")

    nested_auc = roc_auc_score(all_y_outer, all_y_outer_proba)
    nested_f2 = compute_fbeta_from_proba(all_y_outer, all_y_outer_proba)
    common_params = summarize_params(best_params_per_fold)

    return {
        "nested_auc": float(nested_auc),
        "nested_f2": float(nested_f2),
        "outer_fold_aucs": outer_fold_aucs,
        "outer_fold_f2s": outer_fold_f2s,
        "common_params": common_params,
        "best_params_per_fold": best_params_per_fold,
        "all_y_outer": np.array(all_y_outer),
        "all_y_outer_proba": np.array(all_y_outer_proba),
        "nested_shap_payload": nested_shap_payload,
    }


def run_regular_5fold_cv(X_trainval, y_trainval, common_params):
    pipeline = build_pipeline()
    pipeline.set_params(**common_params)

    regular_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    roc_auc_scores = cross_val_score(
        pipeline,
        X_trainval,
        y_trainval,
        cv=regular_cv,
        scoring="roc_auc",
        n_jobs=N_JOBS,
    )
    f2_scores = cross_val_score(
        pipeline,
        X_trainval,
        y_trainval,
        cv=regular_cv,
        scoring=make_scorer(fbeta_score, beta=F_BETA, zero_division=0),
        n_jobs=N_JOBS,
    )

    return {
        "roc_auc_mean": float(np.mean(roc_auc_scores)),
        "roc_auc_std": float(np.std(roc_auc_scores)),
        "f2_mean": float(np.mean(f2_scores)),
        "f2_std": float(np.std(f2_scores)),
    }


def fit_final_model(X_trainval, y_trainval, common_params):
    pipeline = build_pipeline()
    pipeline.set_params(**common_params)
    pipeline.fit(X_trainval, y_trainval)
    return pipeline


def main():
    data = load_data()
    X = data.select_dtypes(include=[np.number]).copy()
    y = data["label"].map({"benign": 0, "malignant": 1})

    X_trainval, _, y_trainval, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    nested_results = run_nested_cv(X_trainval, y_trainval)
    regular_cv_results = run_regular_5fold_cv(
        X_trainval,
        y_trainval,
        nested_results["common_params"],
    )

    print("\n" + "=" * 40)
    print("Summary over outer folds")
    print(f"Nested CV ROC-AUC: {nested_results['nested_auc']:.3f}")
    print(f"Nested CV F{F_BETA}-score: {nested_results['nested_f2']:.3f}")
    print(
        f"5-fold ROC-AUC: {regular_cv_results['roc_auc_mean']:.3f} +/- "
        f"{regular_cv_results['roc_auc_std']:.3f}"
    )
    print(
        f"5-fold F{F_BETA}-score: {regular_cv_results['f2_mean']:.3f} +/- "
        f"{regular_cv_results['f2_std']:.3f}"
    )
    print(f"Most common hyperparameters: {nested_results['common_params']}")

    hyperparameter_counts = {
        key: Counter(params[key] for params in nested_results["best_params_per_fold"])
        for key in nested_results["best_params_per_fold"][0]
    }
    print("Hyperparameter frequencies:")
    for key, counts in hyperparameter_counts.items():
        print(f"{key}: {dict(counts)}")

    run_nested_shap_summary(nested_results["nested_shap_payload"])
    save_roc_curve(
        nested_results["all_y_outer"],
        nested_results["all_y_outer_proba"],
        nested_results["nested_auc"],
    )


if __name__ == "__main__":
    main()


def Logistic_Uni(test=False):
    data = load_data()
    X = data.select_dtypes(include=[np.number]).copy()
    y = data["label"].map({"benign": 0, "malignant": 1})

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    nested_results = run_nested_cv(X_trainval, y_trainval)
    final_pipeline = fit_final_model(
        X_trainval,
        y_trainval,
        nested_results["common_params"],
    )

    run_nested_shap_summary(nested_results["nested_shap_payload"])
    save_roc_curve(
        nested_results["all_y_outer"],
        nested_results["all_y_outer_proba"],
        nested_results["nested_auc"],
    )

    results = {
        "nested_auc_lg": nested_results["nested_auc"],
        "nested_f2_lg": nested_results["nested_f2"],
        "fold_aucs_lg": nested_results["outer_fold_aucs"],
        "outer_fold_f2s_lg": nested_results["outer_fold_f2s"],
        "common_params_lg": nested_results["common_params"],
        "features_per_fold_lg": [X.columns.to_list()] * N_OUTER_SPLITS,
    }
    test_scores = final_pipeline.predict_proba(X_test)[:, 1]

    if test:
        test_auc = roc_auc_score(y_test, test_scores)
        test_pred = (test_scores > 0.5).astype(int)
        test_f2 = fbeta_score(y_test, test_pred, beta=2)

        results["test_auc"] = float(test_auc)
        results["test_f2"] = float(test_f2)
        results["test_scores"] = test_scores
        results["y_test"] = y_test.to_numpy()

    return results
