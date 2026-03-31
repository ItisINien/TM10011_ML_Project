# %% IMPORTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import  VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from worcliver.load_data import load_data
import shap
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.metrics import roc_curve, auc
from collections import Counter
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt


# %% 1 CUSTOM TRANSFORMER VOOR CORR + SELECTK
class CorrAndSelect(BaseEstimator, TransformerMixin):
    def __init__(self, corr_threshold=0.9):
        # self.k = k
        self.corr_threshold = corr_threshold
    
    def fit(self, X, y):
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

    # **Houd alle overgebleven features**
        self.features_ = X_filtered.columns

        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        
        X_const = pd.DataFrame(self.var_thresh_.transform(X),
                               columns=X.columns[self.var_thresh_.get_support()])
        X_filtered = X_const.drop(columns=self.to_drop_, errors='ignore')
        X_selected = X_filtered[self.features_]
        return X_selected
    
    # %% 2 LOAD DATA
data = load_data()
X = data.select_dtypes(include=[np.number])
y = data['label'].map({'benign': 0, 'malignant': 1})

f2_scorer = make_scorer(fbeta_score, beta=2)


# %% 3 TRAIN/TEST SPLIT
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %% 4 NESTED CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

all_y_outer = []
all_y_outer_proba = []

feature_importances_list = []
best_params_list = []
fold_aucs = []

for outer_train_idx, outer_val_idx in outer_cv.split(X_trainval, y_trainval):
    X_outer_train = X_trainval.iloc[outer_train_idx].copy()
    X_outer_val = X_trainval.iloc[outer_val_idx].copy()
    y_outer_train = y_trainval.iloc[outer_train_idx]
    y_outer_val = y_trainval.iloc[outer_val_idx]

    # Pipeline voor inner loop: correlatie + univariate + scaling + classifier
    pipeline = Pipeline([
    ('feat_select', CorrAndSelect(corr_threshold=0.9)),
    ('scaler', RobustScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

    param_dist = {
    'clf__n_estimators': [100, 200, 300, 500],
    'clf__max_depth': [None, 5, 10, 20],
    'clf__min_samples_split': [2, 5, 10]
}

    # Inner loop GridSearchCV
    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=30,           
        cv=inner_cv,
        scoring=f2_scorer,
        n_jobs=1,
        random_state=42)
    
    grid_search.fit(X_outer_train, y_outer_train)

    best_params_list.append(grid_search.best_params_)

    print(f"Outer fold best params: {grid_search.best_params_}")

    # Beste model toepassen op outer validation fold
    best_model = grid_search.best_estimator_
    y_outer_proba = best_model.predict_proba(X_outer_val)[:,1]

    # Print aantal features na const + correlatie
    X_const = pd.DataFrame(
    best_model.named_steps['feat_select'].var_thresh_.transform(X_outer_train),
    columns=best_model.named_steps['feat_select'].const_kept_features_)

    X_corr = X_const.drop(columns=best_model.named_steps['feat_select'].to_drop_, errors="ignore")
    X_selected = X_corr[best_model.named_steps['feat_select'].features_]

    print(
    f"Outer fold: Features orig={X_outer_train.shape[1]}, "
    f"na const={X_const.shape[1]}, "
    f"na corr={X_corr.shape[1]}"
)

    all_y_outer.extend(y_outer_val)
    all_y_outer_proba.extend(y_outer_proba)

    fold_auc = roc_auc_score(y_outer_val, y_outer_proba)
    fold_aucs.append(fold_auc)
    print(f"Outer fold AUC: {fold_auc:.3f}")

    # Feature importances opslaan (alleen geselecteerde features)
    selected_features = best_model.named_steps['feat_select'].features_
    importances = best_model.named_steps['clf'].feature_importances_
    feature_importances_list.append(pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }))

# %% 5 EVALUATE NESTED CV
roc_auc = roc_auc_score(all_y_outer, all_y_outer_proba)
print(f"Nested CV ROC-AUC: {roc_auc:.3f}")

from sklearn.metrics import fbeta_score

# probabilities -> class labels
y_pred = (np.array(all_y_outer_proba) >= 0.5).astype(int)

# F2 score
nested_f2 = fbeta_score(all_y_outer, y_pred, beta=2)
print(f"\nNested CV F2-score: {nested_f2:.3f}")


# %% 7 FINAL MODEL MET BESTE HYPERPARAMETERS VAN NESTED CV

# 1 Kies de beste hyperparameters van de outer folds
best_params = grid_search.best_params_

final_params = {}
for param in best_params_list[0].keys():
    values = [p[param] for p in best_params_list]
    final_params[param] = Counter(values).most_common(1)[0][0]

print("Consensus hyperparameters over alle outer folds:")
print(final_params)

# 2 Maak final pipeline met dezelfde CorrAndSelect config en beste RF params
pipeline_final = Pipeline([
    ('feat_select', CorrAndSelect(corr_threshold=0.9)),
    ('scaler', RobustScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=final_params['clf__n_estimators'],
        max_depth=final_params['clf__max_depth'],
        min_samples_split=final_params['clf__min_samples_split'],
        random_state=42
    ))
])

# 3 Fit pipeline op volledige trainval set
pipeline_final.fit(X_trainval, y_trainval)

# 4 Transformeer test set en predict proba
X_test_transformed = pipeline_final.named_steps['feat_select'].transform(X_test)
y_test_proba = pipeline_final.named_steps['clf'].predict_proba(X_test_transformed)[:,1]

# 7 SHAP ANALYSE OP FINAL MODEL
X_model = pipeline_final.named_steps['feat_select'].transform(X_trainval)
X_model = pd.DataFrame(X_model, columns=pipeline_final.named_steps['feat_select'].features_)

explainer = shap.TreeExplainer(pipeline_final.named_steps['clf'])
shap_values = explainer.shap_values(X_model)

# SHAP array correct afhandelen
if isinstance(shap_values, list):
    shap_vals = np.array(shap_values[1])
elif shap_values.ndim == 3:
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
assert mean_abs_shap.shape[0] == X_model.shape[1], \
    f"Feature mismatch: {mean_abs_shap.shape[0]} vs {X_model.shape[1]}"

shap_importance = pd.DataFrame({
    "feature": X_model.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False)

# Top 20 SHAP features plotten
plt.figure(figsize=(10,6))
plt.barh(shap_importance.head(20).feature[::-1], 
         shap_importance.head(20).mean_abs_shap[::-1])
plt.xlabel("Mean Absolute SHAP Value")
plt.title("Top 20 SHAP Feature Importance")
plt.show()

# 8 Print gekozen hyperparameters
print("\nGekozen hyperparameters RandomForest final model:")
for param, value in pipeline_final.named_steps['clf'].get_params().items():
    print(f"{param}: {value}")

# %%
# %% Nested ROC curve
from sklearn.metrics import roc_curve, auc

# ROC curve berekenen op outer fold predictions
fpr, tpr, thresholds = roc_curve(all_y_outer, all_y_outer_proba)
roc_auc_nested = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"Nested CV ROC (AUC = {roc_auc_nested:.3f})")
plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Nested Cross Validation Cor and Const selection")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.show()


# %%
def Forest_const_cor(test=False):

    results = {
    "nested_auc_const_cor": float(roc_auc),
    "nested_f2_const_cor": float(nested_f2),
    "fold_aucs_const_cor": (fold_aucs),

}

    pipeline_final.fit(X_trainval, y_trainval)

    if test:
        test_scores = pipeline_final.predict_proba(X_test)[:,1]
        test_pred = (test_scores > 0.5).astype(int)
        test_auc = roc_auc_score(y_test, test_scores)
        test_f2 = fbeta_score(y_test, test_pred, beta=2)

        results["test_auc"] = float(test_auc)
        results["test_f2"] = float(test_f2)
        results["y_test"] = y_test 
        results["test_scores"] = test_scores  

    return results
