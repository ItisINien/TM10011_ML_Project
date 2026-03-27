# %% IMPORTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from worcliver.load_data import load_data
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, fbeta_score


# %% 1️⃣ CUSTOM TRANSFORMER VOOR CORR + CONSTANT FILTER
class CorrAndSelect(BaseEstimator, TransformerMixin):
    def __init__(self, corr_threshold=0.9):
        self.corr_threshold = corr_threshold
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        # Constanten verwijderen
        self.var_thresh_ = VarianceThreshold(threshold=0)
        X_const = pd.DataFrame(self.var_thresh_.fit_transform(X), 
                               columns=X.columns[self.var_thresh_.get_support()])
        self.const_kept_features_ = X_const.columns

        # Correlatie filter
        corr_matrix = X_const.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        X_filtered = X_const.drop(columns=self.to_drop_)

        # Bewaar overgebleven features
        self.features_ = X_filtered.columns
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        X_const = pd.DataFrame(self.var_thresh_.transform(X),
                               columns=X.columns[self.var_thresh_.get_support()])
        X_filtered = X_const.drop(columns=self.to_drop_, errors='ignore')
        return X_filtered[self.features_]

# %% 2️⃣ LOAD DATA
data = load_data()
X = data.select_dtypes(include=[np.number])
y = data['label'].map({'benign': 0, 'malignant': 1})

f2_scorer = make_scorer(fbeta_score, beta=2)

# %% 3️⃣ TRAIN/TEST SPLIT
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %% 4️⃣ NESTED CV MET UNIVARIATE SELECTIE
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

all_y_outer = []
all_y_outer_proba = []
feature_importances_list = []
fold_aucs = []

k_univar = 10  # top univariate features

for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(outer_cv.split(X_trainval, y_trainval), 1):
    print(f"\n--- Outer Fold {fold_idx} ---")
    
    X_outer_train = X_trainval.iloc[outer_train_idx].copy()
    X_outer_val = X_trainval.iloc[outer_val_idx].copy()
    y_outer_train = y_trainval.iloc[outer_train_idx]
    y_outer_val = y_trainval.iloc[outer_val_idx]

    # Pipeline: Corr + Univariate + Scaler + Classifier

    pipeline = Pipeline([
        ("feat_select", CorrAndSelect(corr_threshold=0.9)),
        ("scaler", RobustScaler()),
        ("select_k", SelectKBest(score_func=f_classif)),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    param_dist = {
    "select_k__k": [5, 10, 20, 30, 50],
    "clf__n_estimators": [100,200,300,500],
    "clf__max_depth": [None,5,10,20],
    "clf__min_samples_split": [2,5,10]
    }

    # RandomizedSearchCV inner loop
    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=30,  # sneller
        cv=inner_cv,
        scoring=f2_scorer,
        n_jobs=1,
        random_state=42
    )

    grid_search.fit(X_outer_train, y_outer_train)

    best_model = grid_search.best_estimator_
    y_outer_proba = best_model.predict_proba(X_outer_val)[:,1]

    # Feature counts per fold
    X_corr = best_model.named_steps['feat_select'].transform(X_outer_train)
    X_univar = best_model.named_steps['select_k'].transform(X_corr)    
    print(
        f"Features orig={X_outer_train.shape[1]}, "
        f"na const+corr={X_corr.shape[1]}, "
        f"na univar={X_univar.shape[1]}"
    )

    all_y_outer.extend(y_outer_val)
    all_y_outer_proba.extend(y_outer_proba)

    fold_auc = roc_auc_score(y_outer_val, y_outer_proba)
    fold_aucs.append(fold_auc)
    print(f"Outer fold AUC: {fold_auc:.3f}")
    print(f"Best params: {grid_search.best_params_}")

    # Feature importances opslaan
    univar_features = X_corr.columns[best_model.named_steps['select_k'].get_support()]
    importances = best_model.named_steps['clf'].feature_importances_
    feature_importances_list.append(pd.DataFrame({
        'feature': univar_features,
        'importance': importances
    }))

# %% 5️⃣ NESTED CV SCORE
nested_auc = roc_auc_score(all_y_outer, all_y_outer_proba)
print(f"\nNested CV ROC-AUC: {nested_auc:.3f}")

y_pred = (np.array(all_y_outer_proba) >= 0.5).astype(int)

# F2 score
nested_f2 = fbeta_score(all_y_outer, y_pred, beta=2)

print(f"\nNested CV F2-score: {nested_f2:.3f}")



# %%
def Forest_Uni():

    results = {
    "nested_auc_uni": float(nested_auc),
    "nested_f2_uni": float(nested_f2),
    "fold_aucs_uni": float(fold_aucs),
}
    return results