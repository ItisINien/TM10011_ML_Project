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
from sklearn.feature_selection import RFE, SelectKBest, f_classif
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
ROC_OUTPUT_PATH = "roc_curve_nested_cv_lg.png"
MIN_FEATURE_FOLD_COUNT = 3


class CorrAndSelect(BaseEstimator, TransformerMixin):
    def __init__(self, k=200, corr_threshold=0.9):
        self.k = k
        self.corr_threshold = corr_threshold

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.feature_names_in_)
        return pd.DataFrame(X)

    def fit(self, X, y):
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = X_df.columns.to_list()

        constant_mask = X_df.nunique(dropna=False) <= 1
        self.constant_features_ = X_df.columns[constant_mask].to_list()
        self.near_constant_features_ = []
        self.removed_low_variation_features_ = self.constant_features_
        X_non_constant = X_df.drop(
            columns=self.removed_low_variation_features_,
            errors="ignore",
        )

        corr_matrix = X_non_constant.corr(method="spearman").abs().fillna(0)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [
            col for col in upper.columns if any(upper[col] > self.corr_threshold)
        ]

        X_filtered = X_non_constant.drop(columns=self.to_drop_, errors="ignore")

        if X_filtered.shape[1] == 0:
            raise ValueError("No features left after constant/correlation filtering.")

        k_value = min(self.k, X_filtered.shape[1])
        self.selector_ = SelectKBest(score_func=f_classif, k=k_value)
        self.selector_.fit(X_filtered, y)

        self.features_ = X_filtered.columns[self.selector_.get_support()].to_list()
        self.n_input_features_ = X_df.shape[1]
        self.n_after_constant_ = X_non_constant.shape[1]
        self.n_after_correlation_ = X_filtered.shape[1]
        self.n_after_univariate_ = len(self.features_)
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X)
        X_non_constant = X_df.drop(
            columns=self.removed_low_variation_features_,
            errors="ignore",
        )
        X_filtered = X_non_constant.drop(columns=self.to_drop_, errors="ignore")
        transformed = self.selector_.transform(X_filtered)
        return pd.DataFrame(transformed, columns=self.features_, index=X_df.index)


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
    lr_for_rfe = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    final_lr = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            ("feat_select", CorrAndSelect(k=200, corr_threshold=0.9)),
            ("scaler", DataFrameRobustScaler()),
            (
                "rfe",
                RFE(
                    estimator=lr_for_rfe,
                    n_features_to_select=10,
                    step=0.1,
                ),
            ),
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


def summarize_feature_stability(features_per_fold):
    all_features = [feature for fold_features in features_per_fold for feature in fold_features]
    return Counter(all_features)


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
    plt.title("Nested CV ROC Curve (Logistic Regression)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve to: {Path(output_path).resolve()}")


def run_shap_summary(final_pipeline, X_trainval):
    if shap is None:
        print("\nSHAP summary skipped: package 'shap' is not installed.")
        return

    selector = final_pipeline.named_steps["feat_select"]
    scaler = final_pipeline.named_steps["scaler"]
    rfe = final_pipeline.named_steps["rfe"]
    clf = final_pipeline.named_steps["clf"]

    selected_after_uni = selector.transform(X_trainval)
    scaled_after_uni = pd.DataFrame(
        scaler.transform(selected_after_uni),
        columns=selector.features_,
        index=X_trainval.index,
    )

    final_features = np.array(selector.features_)[rfe.support_].tolist()
    scaled_final = pd.DataFrame(
        rfe.transform(scaled_after_uni),
        columns=final_features,
        index=X_trainval.index,
    )

    background_size = min(100, len(scaled_final))
    background = scaled_final.sample(background_size, random_state=RANDOM_STATE)

    explainer = shap.LinearExplainer(clf, background)
    shap_values = explainer(scaled_final)

    importance_df = (
        pd.DataFrame(
            {
                "feature": scaled_final.columns,
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 40)
    print("SHAP mean absolute importance")
    print(importance_df.to_string(index=False))


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
        "feat_select__k": [10, 15, 20, 30],
        "feat_select__corr_threshold": [0.8, 0.85, 0.9],
        "rfe__n_features_to_select": [6, 8, 10, 12],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "rfe__estimator__C": [0.01, 0.1, 1, 10],
    }

    all_y_outer = []
    all_y_outer_proba = []
    outer_fold_aucs = []
    outer_fold_f2s = []
    features_per_fold = []
    best_params_per_fold = []

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
            n_iter=N_RANDOM_SEARCH_ITERATIONS,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
        )
        search.fit(X_outer_train, y_outer_train)

        best_model = search.best_estimator_
        selector = best_model.named_steps["feat_select"]
        rfe = best_model.named_steps["rfe"]

        y_outer_proba = best_model.predict_proba(X_outer_val)[:, 1]
        fold_auc = roc_auc_score(y_outer_val, y_outer_proba)
        fold_f2 = compute_fbeta_from_proba(y_outer_val, y_outer_proba)

        selected_features = np.array(selector.features_)[rfe.support_].tolist()

        all_y_outer.extend(y_outer_val)
        all_y_outer_proba.extend(y_outer_proba)
        outer_fold_aucs.append(fold_auc)
        outer_fold_f2s.append(fold_f2)
        features_per_fold.append(selected_features)
        best_params_per_fold.append(search.best_params_)

        print(f"Best params: {search.best_params_}")
        print(
            "Features per step: "
            f"{selector.n_input_features_} -> "
            f"{selector.n_after_constant_} (after constant) -> "
            f"{selector.n_after_correlation_} (after correlation) -> "
            f"{selector.n_after_univariate_} (after univariate) -> "
            f"{len(selected_features)} (after RFE)"
        )
        print(f"Selected features: {selected_features}")
        print(f"Outer fold ROC-AUC: {fold_auc:.3f}")
        print(f"Outer fold F{F_BETA}-score: {fold_f2:.3f}")

    nested_auc = roc_auc_score(all_y_outer, all_y_outer_proba)
    nested_f2 = compute_fbeta_from_proba(all_y_outer, all_y_outer_proba)
    feature_counts = summarize_feature_stability(features_per_fold)
    common_params = summarize_params(best_params_per_fold)

    return {
        "nested_auc": float(nested_auc),
        "nested_f2": float(nested_f2),
        "outer_fold_aucs": outer_fold_aucs,
        "outer_fold_f2s": outer_fold_f2s,
        "feature_counts": feature_counts,
        "common_params": common_params,
        "best_params_per_fold": best_params_per_fold,
        "features_per_fold": features_per_fold,
        "all_y_outer": np.array(all_y_outer),
        "all_y_outer_proba": np.array(all_y_outer_proba),
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

    print("Most common features:")
    stable_features = [
        (feature, count)
        for feature, count in nested_results["feature_counts"].most_common()
        if count >= MIN_FEATURE_FOLD_COUNT
    ]
    if stable_features:
        for feature, count in stable_features:
            print(f"{feature}: selected in {count}/{N_OUTER_SPLITS} folds")
    else:
        print(f"No features were selected in at least {MIN_FEATURE_FOLD_COUNT}/{N_OUTER_SPLITS} folds.")

    final_pipeline = fit_final_model(
        X_trainval,
        y_trainval,
        nested_results["common_params"],
    )
    run_shap_summary(final_pipeline, X_trainval)
    save_roc_curve(
        nested_results["all_y_outer"],
        nested_results["all_y_outer_proba"],
        nested_results["nested_auc"],
    )


if __name__ == "__main__":
    main()
