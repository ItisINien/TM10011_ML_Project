# %%
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from worcliver.load_data import load_data
try:
    import shap
except ImportError:
    shap = None


RANDOM_STATE = 42


class CorrUnivariateSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        corr_threshold=0.9,
        k_univariate=20,
        consensus_n_splits=5,
        consensus_min_fraction=0.6,
    ):
        self.corr_threshold = corr_threshold
        self.k_univariate = k_univariate
        self.consensus_n_splits = consensus_n_splits
        self.consensus_min_fraction = consensus_min_fraction

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()

        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.feature_names_in_)

        return pd.DataFrame(X)

    def _select_features_once(self, X_df, y):
        constant_mask = X_df.nunique(dropna=False) <= 1
        constant_features = X_df.columns[constant_mask].to_list()
        X_non_constant = X_df.drop(columns=constant_features, errors="ignore")

        if X_non_constant.shape[1] == 0:
            return [], constant_features, []

        corr_matrix = X_non_constant.corr(method="spearman").abs().fillna(0)
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        correlation_features = [
            col for col in upper.columns if any(upper[col] > self.corr_threshold)
        ]
        X_corr = X_non_constant.drop(columns=correlation_features, errors="ignore")

        if X_corr.shape[1] == 0:
            return [], constant_features, correlation_features

        k = min(self.k_univariate, X_corr.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_corr, y)
        selected_features = X_corr.columns[selector.get_support()].to_list()
        return selected_features, constant_features, correlation_features

    def fit(self, X, y):
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = X_df.columns.to_list()

        y_series = pd.Series(y, index=X_df.index)
        final_features, constant_features, correlation_features = self._select_features_once(
            X_df, y_series
        )
        self.constant_features_ = constant_features
        self.correlation_features_ = correlation_features
        self.univariate_features_ = final_features.copy()

        if len(final_features) == 0:
            raise ValueError("No features left after constant, correlation, and univariate filtering.")

        consensus_cv = StratifiedKFold(
            n_splits=self.consensus_n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        features_per_consensus_fold = []
        for train_idx, _ in consensus_cv.split(X_df, y_series):
            X_fold = X_df.iloc[train_idx]
            y_fold = y_series.iloc[train_idx]
            fold_features, _, _ = self._select_features_once(X_fold, y_fold)
            if fold_features:
                features_per_consensus_fold.append(fold_features)

        if not features_per_consensus_fold:
            self.features_ = final_features.copy()
            self.consensus_feature_counts_ = Counter(final_features)
            return self

        feature_counts = Counter(
            feature
            for fold_features in features_per_consensus_fold
            for feature in fold_features
        )
        min_count = max(
            1,
            int(np.ceil(self.consensus_min_fraction * len(features_per_consensus_fold))),
        )
        consensus_features = [
            feature for feature, count in feature_counts.items() if count >= min_count
        ]

        if not consensus_features:
            consensus_features = [
                feature
                for feature, _ in feature_counts.most_common(min(len(final_features), 10))
            ]

        ordered_consensus = [
            feature for feature in final_features if feature in set(consensus_features)
        ]
        self.features_ = ordered_consensus if ordered_consensus else final_features.copy()
        self.consensus_feature_counts_ = feature_counts
        self.consensus_min_count_ = min_count
        self.consensus_features_per_fold_ = features_per_consensus_fold
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X)
        return X_df[self.features_]


def build_pipeline():
    return Pipeline(
        [
            (
                "feat_select",
                CorrUnivariateSelector(
                    corr_threshold=0.9,
                    k_univariate=20,
                    consensus_n_splits=5,
                    consensus_min_fraction=0.6,
                ),
            ),
            ("scaler", RobustScaler()),
            ("clf", SVC(kernel="linear")),
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
        "feat_select__k_univariate": [10, 15],  # Aantal features dat na ANOVA univariate selectie overblijft.
        "feat_select__consensus_min_fraction": [0.6, 0.8],  # Alleen features behouden die in meerdere consensus-folds terugkomen.
        "clf__C": [0.01, 0.1, 1, 10],  # Lage tot matig hoge C houdt regularisatie aanwezig, maar geeft iets meer ruimte dan de strengste setting.
    }

    all_y_outer = []
    all_scores_outer = []
    features_per_fold = []
    best_params_per_fold = []
    all_fold_scores = []
    all_fold_y = []

    f2_score = make_scorer(fbeta_score, beta=2)

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
            scoring=f2_score,
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
            f"{len(selector.features_)} (after consensus)"
        )
        print(f"Selected features: {selector.features_}")
        print(
            f"Outer fold ROC-AUC: "
            f"{roc_auc_score(y_outer_val, scores_outer):.3f}"
        )
    # na het berekenen van scores_outer
    all_y_outer.extend(y_outer_val)
    all_scores_outer.extend(scores_outer)

    # Voeg per fold toe
    all_fold_scores.append(scores_outer)
    all_fold_y.append(y_outer_val)

    nested_auc = roc_auc_score(all_y_outer, all_scores_outer)
    feature_counts, consensus_features = summarize_feature_stability(
        features_per_fold, outer_cv.get_n_splits()
    )
    final_params = summarize_params(best_params_per_fold)
    pred_labels = np.concatenate([ (s > 0).astype(int) for s in all_fold_scores ])
    true_labels = np.concatenate([ y.values for y in all_fold_y ])
    nested_f2 = fbeta_score(true_labels, pred_labels, beta=2)
    fold_aucs = []
    fold_auc = roc_auc_score(y_outer_val, scores_outer)
    fold_aucs.append(fold_auc)

    return nested_auc, nested_f2, fold_aucs, feature_counts, consensus_features, final_params

def main():
    data = load_data()
    X = data.select_dtypes(include=[np.number]).copy()
    y = data["label"].map({"benign": 0, "malignant": 1}).astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    nested_auc, nested_f2, fold_aucs, feature_counts, consensus_features, final_params = run_nested_cv(X_trainval, y_trainval)


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




# %%
def SVM_Uni():
    # 1. Laad de data
    data = load_data()
    X = data.select_dtypes(include=[np.number]).copy()
    y = data["label"].map({"benign": 0, "malignant": 1}).astype(int)

    # 2. Train/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 3. Run nested CV en pak alleen de eerste 3 resultaten
    nested_auc, nested_f2, fold_aucs, *_ = run_nested_cv(X_trainval, y_trainval)

    # 4. Stop resultaten in dictionary
    results = {
        "nested_auc_SVM_uni": float(nested_auc),
        "nested_f2_SVM_uni": float(nested_f2),
        "fold_aucs_SVM_uni": fold_aucs,
    }
    return results

# %%
if __name__ == "__main__":
    main()