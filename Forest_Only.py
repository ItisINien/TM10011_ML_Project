# %% IMPORTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from worcliver.load_data import load_data
import shap
from sklearn.metrics import make_scorer, fbeta_score
import matplotlib.pyplot as plt
    
# %%  Load data and define F2
data = load_data()
X = data.select_dtypes(include=[np.number])
y = data['label'].map({'benign': 0, 'malignant': 1})

f2_scorer = make_scorer(fbeta_score, beta=2)


# %% train/test split (80/20)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %% Nested CV and Grindsearch
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

all_y_outer = []
all_y_outer_proba = []

feature_importances_list = []
fold_aucs = []

for outer_train_idx, outer_val_idx in outer_cv.split(X_trainval, y_trainval):
    X_outer_train = X_trainval.iloc[outer_train_idx].copy()
    X_outer_val = X_trainval.iloc[outer_val_idx].copy()
    y_outer_train = y_trainval.iloc[outer_train_idx]
    y_outer_val = y_trainval.iloc[outer_val_idx]

    # Pipeline voor inner loop: correlatie + univariate + scaling + classifier
    pipeline = Pipeline([
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
        n_iter=100, 
        scoring=f2_scorer,
        n_jobs=-1,
        random_state=42)
    grid_search.fit(X_outer_train, y_outer_train)

    print(f"Outer fold best params: {grid_search.best_params_}")
    print(f"Outer fold best inner CV ROC-AUC: {grid_search.best_score_:.3f}")

    # Beste model on outer validation fold
    best_model = grid_search.best_estimator_
    y_outer_proba = best_model.predict_proba(X_outer_val)[:,1]

    all_y_outer.extend(y_outer_val)
    all_y_outer_proba.extend(y_outer_proba)

    fold_auc = roc_auc_score(y_outer_val, y_outer_proba)
    fold_aucs.append(fold_auc)
    print(f"Outer fold AUC: {fold_auc:.3f}")

    selected_features = X_outer_train.columns
    importances = best_model.named_steps['clf'].feature_importances_

    feature_importances_list.append(pd.DataFrame({
        'feature': selected_features,
        'importance': importances
        }))

# %% Evaluation nested CV with ROC_AUC and F2

#ROC-AUC
roc_auc = roc_auc_score(all_y_outer, all_y_outer_proba)
print(f"Nested CV ROC-AUC: {roc_auc:.3f}")

# probabilities -> class labels
y_pred = (np.array(all_y_outer_proba) >= 0.5).astype(int)

# F2 score
nested_f2 = fbeta_score(all_y_outer, y_pred, beta=2)
print(f"\nNested CV F2-score: {nested_f2:.3f}")

# %% SHAP
X_model = X_outer_train.copy()

explainer = shap.TreeExplainer(best_model.named_steps['clf'])

#SHAP value per feature per sample
shap_values = explainer.shap_values(X_model)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    # Older SHAP: list van arrays per klasse →  klasse 1
    shap_vals = np.array(shap_values[1])
elif shap_values.ndim == 3:
    # Newer SHAP: (n_samples, n_features, n_classes) → take klasse 1
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

print("SHAP array shape:", shap_vals.shape)  # (n_samples, n_features)

# Mean absolute SHAP per feature → 1D
mean_abs_shap = np.abs(shap_vals).mean(axis=0)

# Check amount of features
assert mean_abs_shap.shape[0] == X_model.shape[1], \
    f"Feature mismatch: {mean_abs_shap.shape[0]} vs {X_model.shape[1]}"

# DataFrame 
shap_importance = pd.DataFrame({
    "feature": X_model.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False)

# Print top 20
print(shap_importance.head(20))

# Plot top 20
plt.figure(figsize=(10,6))
plt.barh(shap_importance.head(20).feature[::-1], 
         shap_importance.head(20).mean_abs_shap[::-1])
plt.xlabel("Mean Absolute SHAP Value")
plt.title("Top 20 SHAP Feature Importance")
plt.show()

# %%
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_curve, auc

# ROC curve van nested CV predictions
fpr, tpr, thresholds = roc_curve(all_y_outer, all_y_outer_proba)
roc_auc_nested = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"Forest Nested CV ROC (AUC = {roc_auc_nested:.3f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Nested Cross Validation no Feature Selection")
plt.legend(loc="lower right")
plt.show()

# %% FUNCTION RESULTS
def Forest_Only_results(test=False):
    results = {
        "nested_auc_only": float(roc_auc),
        "nested_f2_only": float(nested_f2),
        "fold_aucs_only": fold_aucs,  
    }

    # Train final model on full trainval set
    best_model.fit(X_trainval, y_trainval)

# Predict on test set
    test_scores = best_model.predict_proba(X_test)[:,1]

    test_pred = (test_scores > 0.5).astype(int)    

    if test:
        test_auc = roc_auc_score(y_test, test_scores)
        test_pred = (test_scores > 0.5).astype(int)  # juiste threshold
        test_f2 = fbeta_score(y_test, test_pred, beta=2)

        results["test_auc"] = float(test_auc)
        results["test_f2"] = float(test_f2)
    return results