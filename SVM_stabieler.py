# %%
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from worcliver.load_data import load_data

RANDOM_STATE = 42


class CorrUnivariateSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector:
    1️⃣ verwijdert constant features
    2️⃣ verwijdert sterk correlerende features (Spearman)
    3️⃣ pakt top-k features via univariate ANOVA
    """

    def __init__(self, corr_threshold=0.9, k_univariate=10):
        self.corr_threshold = corr_threshold
        self.k_univariate = k_univariate

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.feature_names_in_)
        return pd.DataFrame(X)

    def fit(self, X, y):
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = X_df.columns.to_list()

        # 1️⃣ Verwijder constant features
        constant_mask = X_df.nunique(dropna=False) <= 1
        self.constant_features_ = X_df.columns[constant_mask].to_list()
        X_non_constant = X_df.drop(columns=self.constant_features_, errors="ignore")

        # 2️⃣ Verwijder sterk correlerende features
        corr_matrix = X_non_constant.corr(method="spearman").abs().fillna(0)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.correlation_features_ = [
            col for col in upper.columns if any(upper[col] > self.corr_threshold)
        ]
        X_corr = X_non_constant.drop(columns=self.correlation_features_, errors="ignore")

        # 3️⃣ Top-k univariate features
        k = min(self.k_univariate, X_corr.shape[1])
        self.univariate_selector_ = SelectKBest(score_func=f_classif, k=k)
        self.univariate_selector_.fit(X_corr, y)
        self.features_ = X_corr.columns[self.univariate_selector_.get_support()].to_list()

        return self

    def transform(self, X):
        X_df = self._to_dataframe(X)
        X_non_constant = X_df.drop(columns=self.constant_features_, errors="ignore")
        X_corr = X_non_constant.drop(columns=self.correlation_features_, errors="ignore")
        X_uni = X_corr[self.features_]
        return X_uni


def build_pipeline(k_univariate=10, corr_threshold=0.9):
    return Pipeline([
        ("feat_select", CorrUnivariateSelector(corr_threshold=corr_threshold, k_univariate=k_univariate)),
        ("scaler", RobustScaler()),
        ("clf", SVC(probability=True))
    ])


def summarize_params(best_params_per_fold):
    summary = {}
    for key in best_params_per_fold[0]:
        values = [params[key] for params in best_params_per_fold]
        summary[key] = Counter(values).most_common(1)[0][0]
    return summary


def summarize_feature_stability(features_per_fold, n_folds):
    all_selected = [feature for fold in features_per_fold for feature in fold]
    feature_counts = Counter(all_selected)

    consensus_features = [feature for feature, count in feature_counts.items() if count == n_folds]
    if not consensus_features:
        consensus_features = [feature for feature, count in feature_counts.items() if count >= n_folds - 1]

    if not consensus_features:
        consensus_features = [feature for feature, _ in feature_counts.most_common(
            min(10, max(len(features_per_fold[0]), 1))
        )]

    return feature_counts, consensus_features


def run_nested_cv(X_trainval, y_trainval):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    param_dist = {
        "feat_select__k_univariate": [5, 10, 15, 20],
        "feat_select__corr_threshold": [0.8, 0.85, 0.9],
        "clf__kernel": ["linear", "rbf"],
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.001],
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
            n_iter=20,
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

        scores_outer = best_model.predict_proba(X_outer_val)[:,1]
        all_y_outer.extend(y_outer_val)
        all_scores_outer.extend(scores_outer)

        print(f"Best params: {search.best_params_}")
        print(f"Selected features: {selector.features_}")
        print(f"Outer fold ROC-AUC: {roc_auc_score(y_outer_val, scores_outer):.3f}")

    nested_auc = roc_auc_score(all_y_outer, all_scores_outer)
    feature_counts, consensus_features = summarize_feature_stability(features_per_fold, outer_cv.get_n_splits())
    final_params = summarize_params(best_params_per_fold)

    return nested_auc, feature_counts, consensus_features, final_params


def main():
    data = load_data()
    X = data.select_dtypes(include=[np.number]).copy()
    y = data["label"].map({"benign": 0, "malignant": 1})

    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # -----------------------------
    # Nested CV
    # -----------------------------
    nested_auc, feature_counts, consensus_features, final_params = run_nested_cv(X_trainval, y_trainval)

    print("\n" + "=" * 40)
    print("Feature stability")
    print(f"Unique selected features: {len(feature_counts)}")
    print(f"Consensus/stable features: {consensus_features}")
    print("\n" + "=" * 40)
    print(f"Nested CV ROC-AUC: {nested_auc:.3f}")
    print(f"Most common best params over outer folds: {final_params}")

#     # -----------------------------
#     # Final pipeline
#     # -----------------------------
#     final_pipeline = build_pipeline()
#     final_pipeline.set_params(**final_params)

#     # Print hoeveel features we trainen
#     k_final = final_params.get(
#         "feat_select__k_univariate",
#         final_pipeline.named_steps["feat_select"].k_univariate
#     )
#     print(f"\nFinal training: using top k={k_final} univariate features per fold")

#     # 5-fold CV op trainval
#     regular_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
#     regular_cv_scores = cross_val_score(
#         final_pipeline,
#         X_trainval,
#         y_trainval,
#         cv=regular_cv,
#         scoring="roc_auc",
#         n_jobs=-1,
#     )
#     print(f"5-fold CV ROC-AUC: {regular_cv_scores.mean():.3f} +/- {regular_cv_scores.std():.3f}")

#     # Fit op gehele trainval set
#     final_pipeline.fit(X_trainval, y_trainval)

#     # Selecteer features die daadwerkelijk gebruikt worden
#     final_selected_features = final_pipeline.named_steps["feat_select"].features_
#     print(f"\nTraining with {len(final_selected_features)} features:")
#     print(final_selected_features)

#     # Test set evaluatie
#     test_scores = final_pipeline.predict_proba(X_test)[:, 1]
#     test_auc = roc_auc_score(y_test, test_scores)
#     print(f"\nFinal test ROC-AUC: {test_auc:.3f}")


if __name__ == "__main__":
     main()