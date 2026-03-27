# %% IMPORTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer, fbeta_score
from sklearn.base import BaseEstimator, TransformerMixin
from worcliver.load_data import load_data
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, fbeta_score


# %% CUSTOM TRANSFORMER (CONST + CORRELATIE)
class CorrAndSelect(BaseEstimator, TransformerMixin):

    def __init__(self, corr_threshold=0.9):
        self.corr_threshold = corr_threshold
    
    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Constant features verwijderen
        self.var_thresh_ = VarianceThreshold(threshold=0)

        X_const = pd.DataFrame(
            self.var_thresh_.fit_transform(X),
            columns=X.columns[self.var_thresh_.get_support()]
        )

        # Correlatie filter
        corr_matrix = X_const.corr(method="spearman").abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        self.to_drop_ = [
            col for col in upper.columns
            if any(upper[col] > self.corr_threshold)
        ]

        self.features_ = X_const.drop(columns=self.to_drop_).columns

        return self


    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_const = pd.DataFrame(
            self.var_thresh_.transform(X),
            columns=X.columns[self.var_thresh_.get_support()]
        )

        X_filtered = X_const.drop(columns=self.to_drop_, errors="ignore")

        return X_filtered[self.features_]


# %% LOAD DATA
data = load_data()

X = data.select_dtypes(include=[np.number])
y = data["label"].map({"benign": 0, "malignant": 1})

f2_scorer = make_scorer(fbeta_score, beta=2)


# %% TRAIN TEST SPLIT
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# %% NESTED CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

all_y_outer = []
all_y_outer_proba = []

feature_importances_list = []
n_features_rfe = 10
fold_aucs = []


for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(outer_cv.split(X_trainval, y_trainval), 1):

    print(f"\n--- Outer Fold {fold_idx} ---")

    X_outer_train = X_trainval.iloc[outer_train_idx]
    X_outer_val = X_trainval.iloc[outer_val_idx]

    y_outer_train = y_trainval.iloc[outer_train_idx]
    y_outer_val = y_trainval.iloc[outer_val_idx]


    pipeline = Pipeline([
        ("feat_select", CorrAndSelect(corr_threshold=0.9)),
        ("scaler", RobustScaler()),

        ("rfe", RFE(
            estimator=RandomForestClassifier(n_estimators=200, random_state=42),
            n_features_to_select=n_features_rfe
        )),

        ("clf", RandomForestClassifier(random_state=42))
    ])


    param_dist = {

        "clf__n_estimators": [100,200,300,500],
        "clf__max_depth": [None,5,10,20],
        "clf__min_samples_split": [2,5,10]
    }

    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=5,
        cv=inner_cv,
        scoring=f2_scorer,
        n_jobs=1,
        random_state=42
    )

    grid_search.fit(X_outer_train, y_outer_train)

    best_model = grid_search.best_estimator_

    y_outer_proba = best_model.predict_proba(X_outer_val)[:,1]

    X_corr = best_model.named_steps["feat_select"].transform(X_outer_train)

    X_rfe = best_model.named_steps["rfe"].transform(X_corr)


    print(
        f"Features orig={X_outer_train.shape[1]}, "
        f"na const+corr={X_corr.shape[1]}, "
        f"na RFE={X_rfe.shape[1]}"
    )

    fold_auc = roc_auc_score(y_outer_val, y_outer_proba)
    fold_aucs.append(fold_auc)
    print(f"Outer fold AUC: {fold_auc:.3f}")

    print(f"Best params: {grid_search.best_params_}")


    all_y_outer.extend(y_outer_val)

    all_y_outer_proba.extend(y_outer_proba)


# %% NESTED CV PERFORMANCE

nested_auc = roc_auc_score(all_y_outer, all_y_outer_proba)

print(f"\nNested CV ROC-AUC: {nested_auc:.3f}")


y_pred = (np.array(all_y_outer_proba) >= 0.5).astype(int)

nested_f2 = fbeta_score(all_y_outer, y_pred, beta=2)

print(f"\nNested CV F2-score: {nested_f2:.3f}")

# %%
# Transformeer features zoals gebruikt in model
X_model = best_model.named_steps['feat_select'].transform(X_outer_train)
X_model = pd.DataFrame(X_model, columns=best_model.named_steps['feat_select'].features_)

explainer = shap.TreeExplainer(best_model.named_steps['clf'])
shap_values = explainer.shap_values(X_model)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    # Older SHAP: list of arrays per class → take class 1
    shap_vals = np.array(shap_values[1])
elif shap_values.ndim == 3:
    # Newer SHAP: (n_samples, n_features, n_classes) → take class 1
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

print("SHAP array shape:", shap_vals.shape)  # Should be (n_samples, n_features)


# Gemiddelde absolute SHAP per feature → 1D
mean_abs_shap = np.abs(shap_vals).mean(axis=0)

# Controleer dat het aantal features klopt
assert mean_abs_shap.shape[0] == X_model.shape[1], \
    f"Feature mismatch: {mean_abs_shap.shape[0]} vs {X_model.shape[1]}"

# DataFrame maken en sorteren
shap_importance = pd.DataFrame({
    "feature": X_model.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False)

# Print top 20
print(shap_importance.head(20))

# Plot top 20
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.barh(shap_importance.head(20).feature[::-1], 
         shap_importance.head(20).mean_abs_shap[::-1])
plt.xlabel("Mean Absolute SHAP Value")
plt.title("Top 20 SHAP Feature Importance")
plt.show()

# %%
def Forest_opt():

    results = {
    "nested_auc_forest_opt": float(nested_auc),
    "nested_f2_forest_opt": float(nested_f2),
    "fold_aucs_forest_opt": (fold_aucs),
}
    return results