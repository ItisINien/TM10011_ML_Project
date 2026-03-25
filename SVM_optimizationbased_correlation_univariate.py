import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC, SVC

from worcliver.load_data import load_data

try:
    import shap
except ImportError:
    shap = None


RANDOM_STATE = 42


class CorrUnivariateOptimizationSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        corr_threshold=0.9,
        k_univariate=20,
        n_features_to_select=5,
        rfe_estimator_c=1.0,
        rfe_step=1,
    ):
        self.corr_threshold = corr_threshold
        self.k_univariate = k_univariate
        self.n_features_to_select = n_features_to_select
        self.rfe_estimator_c = rfe_estimator_c
        self.rfe_step = rfe_step

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
        X_non_constant = X_df.drop(columns=self.constant_features_, errors="ignore")

        corr_matrix = X_non_constant.corr(method="spearman").abs().fillna(0)
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.correlation_features_ = [
            col for col in upper.columns if any(upper[col] > self.corr_threshold)
        ]
        X_corr = X_non_constant.drop(columns=self.correlation_features_, errors="ignore")

        if X_corr.shape[1] == 0:
            raise ValueError("No features left after constant and correlation filtering.")

        k = min(self.k_univariate, X_corr.shape[1])
        self.univariate_selector_ = SelectKBest(score_func=f_classif, k=k)
        self.univariate_selector_.fit(X_corr, y)

        self.univariate_features_ = X_corr.columns[
            self.univariate_selector_.get_support()
        ].to_list()
        X_uni = X_corr[self.univariate_features_]

        n_rfe = min(self.n_features_to_select, X_uni.shape[1])
        if n_rfe == 0:
            raise ValueError("No features left for optimization-based feature selection.")

        self.rfe_estimator_ = LinearSVC(
            C=self.rfe_estimator_c,
            dual=False,
            max_iter=10000,  # Hoge bovengrens om convergence warnings te voorkomen.
            random_state=RANDOM_STATE,
        )
        self.rfe_selector_ = RFE(
            estimator=self.rfe_estimator_,  # RFE verwijdert stapsgewijs de minst belangrijke features.
            n_features_to_select=n_rfe,
            step=self.rfe_step,
        )
        self.rfe_selector_.fit(X_uni, y)

        self.features_ = X_uni.columns[self.rfe_selector_.get_support()].to_list()
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X)
        X_non_constant = X_df.drop(columns=self.constant_features_, errors="ignore")
        X_corr = X_non_constant.drop(columns=self.correlation_features_, errors="ignore")
        X_uni = X_corr[self.univariate_features_]
        return X_uni[self.features_]


def build_pipeline():
    return Pipeline(
        [
            (
                "feat_select",
                CorrUnivariateOptimizationSelector(
                    corr_threshold=0.9,
                    k_univariate=20,
                    n_features_to_select=5,
                    rfe_estimator_c=1.0,
                ),
            ),
            ("scaler", RobustScaler()),
            ("clf", SVC()),
        ]
    )


def summarize_params(best_params_per_fold):
    summary = {}
    for key in best_params_per_fold[0]:
        values = [params[key] for params in best_params_per_fold]
        summary[key] = Counter(values).most_common(1)[0][0]
    return summary


def summarize_feature_stability(features_per_fold, n_folds):
    all_selected = [feature for fold in features_per_fold for feature in fold]
    feature_counts = Counter(all_selected)

    consensus_features = [
        feature for feature, count in feature_counts.items() if count == n_folds
    ]
    if not consensus_features:
        consensus_features = [
            feature for feature, count in feature_counts.items() if count >= n_folds - 1
        ]

    if not consensus_features:
        consensus_features = [
            feature
            for feature, _ in feature_counts.most_common(
                min(10, max(len(features_per_fold[0]), 1))
            )
        ]

    return feature_counts, consensus_features


def run_shap_analysis(final_pipeline, X_trainval, X_test, output_dir="shap_outputs"):
    if shap is None:
        print("\nSHAP analysis skipped: package 'shap' is not installed.")
        print("Install first with: conda install -c conda-forge shap")
        return

    clf = final_pipeline.named_steps["clf"]
    if clf.kernel != "linear":
        print("\nSHAP analysis skipped: current final classifier is not linear.")
        return

    selector = final_pipeline.named_steps["feat_select"]
    scaler = final_pipeline.named_steps["scaler"]
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # SHAP moet dezelfde geselecteerde features zien als de classifier.
    train_selected = pd.DataFrame(
        selector.transform(X_trainval),
        columns=selector.features_,
        index=X_trainval.index,
    )
    test_selected = pd.DataFrame(
        selector.transform(X_test),
        columns=selector.features_,
        index=X_test.index,
    )

    # We gebruiken ook dezelfde scaling als in de getrainde pipeline.
    train_scaled = pd.DataFrame(
        scaler.transform(train_selected),
        columns=selector.features_,
        index=X_trainval.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_selected),
        columns=selector.features_,
        index=X_test.index,
    )

    # Kleine achtergrondset houdt de SHAP-berekening sneller en stabiel.
    background_size = min(100, len(train_scaled))
    background = train_scaled.sample(background_size, random_state=RANDOM_STATE)

    explainer = shap.LinearExplainer(clf, background)
    shap_values = explainer(test_scaled)

    importance_df = (
        pd.DataFrame(
            {
                "feature": test_scaled.columns,
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 40)
    print("SHAP mean absolute importance")
    print(importance_df.to_string(index=False))

    shap.plots.beeswarm(shap_values, max_display=len(test_scaled.columns), show=False)
    plt.tight_layout()
    plt.savefig(output_path / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.plots.bar(shap_values, max_display=len(test_scaled.columns), show=False)
    plt.tight_layout()
    plt.savefig(output_path / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP plots to: {output_path.resolve()}")


def save_roc_curve(y_true, y_scores, roc_auc, output_path="roc_curve_final_test.png"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final Test ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve to: {Path(output_path).resolve()}")


def run_nested_cv(X_trainval, y_trainval):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    param_dist = {
        "feat_select__corr_threshold": [0.8, 0.85, 0.9],  # Drempel voor wanneer twee features te sterk correleren.
        "feat_select__k_univariate": [10, 15, 20, 30],  # Aantal features dat na ANOVA univariate selectie overblijft.
        "feat_select__n_features_to_select": [3, 5, 8, 10],  # Aantal features dat RFE uiteindelijk behoudt.
        "feat_select__rfe_estimator_c": [0.1, 1, 10],  # C-waarde van de lineaire SVM binnen RFE. LageS C geeft meer robuust model, hoge C zo min mogelijk fouten
        "clf__kernel": ["linear", "rbf"],  # Type scheidingsgrens van de uiteindelijke SVM.
        "clf__C": [0.1, 1, 10, 100],  # Stuurt de balans tussen regularisatie en trainingsfouten. Hoge C geeft SVM kans complexere grens om fouten te verminderen, lage C accepteerd SVM met meer fouten maar vermindert overfitting
        "clf__gamma": ["scale", 0.1, 0.01, 0.001],  # Relevante instelling voor o.a. de RBF-kernel; bepaalt hoe lokaal de invloed van punten is.
    }

    all_y_outer = []
    all_scores_outer = []
    features_per_fold = []
    best_params_per_fold = []

    print("Start nested cross-validation...")

    for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(
        outer_cv.split(X_trainval, y_trainval), start=1
    ):
        print(f"\n--- Outer fold {fold_idx} ---")

        X_outer_train = X_trainval.iloc[outer_train_idx].copy()
        X_outer_val = X_trainval.iloc[outer_val_idx].copy()
        y_outer_train = y_trainval.iloc[outer_train_idx]
        y_outer_val = y_trainval.iloc[outer_val_idx]

        search = RandomizedSearchCV(
            estimator=build_pipeline(),
            param_distributions=param_dist,
            n_iter=20,  # Test 20 willekeurige hyperparametercombinaties per outer fold.
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        search.fit(X_outer_train, y_outer_train)

        best_model = search.best_estimator_
        selector = best_model.named_steps["feat_select"]

        best_params_per_fold.append(search.best_params_)
        features_per_fold.append(selector.features_)

        scores_outer = best_model.decision_function(X_outer_val)
        all_y_outer.extend(y_outer_val)
        all_scores_outer.extend(scores_outer)

        print(f"Best params: {search.best_params_}")
        print(
            "Features: "
            f"{X_outer_train.shape[1]} -> "
            f"{X_outer_train.shape[1] - len(selector.constant_features_)} (after constant) -> "
            f"{X_outer_train.shape[1] - len(selector.constant_features_) - len(selector.correlation_features_)} (after correlation) -> "
            f"{len(selector.univariate_features_)} (after univariate) -> "
            f"{len(selector.features_)} (after optimization)"
        )
        print(f"Selected features: {selector.features_}")
        print(
            f"Outer fold ROC-AUC: "
            f"{roc_auc_score(y_outer_val, scores_outer):.3f}"
        )

    nested_auc = roc_auc_score(all_y_outer, all_scores_outer)
    feature_counts, consensus_features = summarize_feature_stability(
        features_per_fold, outer_cv.get_n_splits()
    )
    final_params = summarize_params(best_params_per_fold)

    return nested_auc, feature_counts, consensus_features, final_params


def main():
    data = load_data()
    X = data.select_dtypes(include=[np.number]).copy()
    y = data["label"].map({"benign": 0, "malignant": 1})

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    nested_auc, feature_counts, consensus_features, final_params = run_nested_cv(
        X_trainval, y_trainval
    )

    print("\n" + "=" * 40)
    print("Feature stability")
    print(f"Unique selected features: {len(feature_counts)}")
    print(f"Consensus/stable features: {consensus_features}")

    print("\n" + "=" * 40)
    print(f"Nested CV ROC-AUC: {nested_auc:.3f}")
    print(f"Most common best params over outer folds: {final_params}")

    final_pipeline = build_pipeline()
    final_pipeline.set_params(**final_params)

    regular_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    regular_cv_scores = cross_val_score(
        final_pipeline,
        X_trainval,
        y_trainval,
        cv=regular_cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    print("\n" + "=" * 40)
    print(
        f"5-fold CV ROC-AUC: {regular_cv_scores.mean():.3f} "
        f"+/- {regular_cv_scores.std():.3f}"
    )

    final_pipeline.fit(X_trainval, y_trainval)
    test_scores = final_pipeline.decision_function(X_test)
    test_auc = roc_auc_score(y_test, test_scores)

    final_selected_features = final_pipeline.named_steps["feat_select"].features_
    final_univariate_features = (
        final_pipeline.named_steps["feat_select"].univariate_features_
    )

    print("\n" + "=" * 40)
    print(f"Final test ROC-AUC: {test_auc:.3f}")
    print(f"Final univariate features ({len(final_univariate_features)}):")
    print(final_univariate_features)
    print(f"Final selected features ({len(final_selected_features)}):")
    print(final_selected_features)
    save_roc_curve(y_test, test_scores, test_auc)
    run_shap_analysis(final_pipeline, X_trainval, X_test)

if __name__ == "__main__":
    main()
