# %% 1️⃣ IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.base import BaseEstimator, TransformerMixin
from worcliver.load_data import load_data

# %% 2️⃣ CUSTOM TRANSFORMERS
class CorrFilter(BaseEstimator, TransformerMixin):
    """Alleen de correlatie-check"""
    def __init__(self, threshold=0.9):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        corr_matrix = X.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]
        return self
    
    def transform(self, X):
        return X.drop(columns=self.to_drop_)

class CorrAndSelect(BaseEstimator, TransformerMixin):
    """De combinatie: Correlatie + Univariate"""
    def __init__(self, k=100, threshold=0.9):
        self.k = k
        self.threshold = threshold
    
    def fit(self, X, y):
        # Correlatie
        corr_matrix = X.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]
        X_filtered = X.drop(columns=self.to_drop_)
        # Univariate
        self.selector_ = SelectKBest(score_func=f_classif, k=min(self.k, X_filtered.shape[1]))
        self.selector_.fit(X_filtered, y)
        return self
    
    def transform(self, X):
        X_filtered = X.drop(columns=self.to_drop_)
        return self.selector_.transform(X_filtered)

# %% 3️⃣ DATA & CV SETUP
data = load_data()
X = data.select_dtypes(include=[np.number])
y = data['label'].map({'benign': 0, 'malignant': 1})

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# %% 4️⃣ HET EXPERIMENT: SVM MET VARIABELE SELECTIE
all_y_val = []
all_probs = []
winning_methods = []

# Hier definiëren we de drie verschillende "paden" voor de SVM
param_grid = [
    {
        'selector': [CorrFilter()],
        'clf__C': [0.1, 1, 10],
        'clf__gamma': ['scale', 0.01]
    },
    {
        'selector': [CorrAndSelect()],
        'selector__k': [50, 100, 150],
        'clf__C': [0.1, 1, 10],
        'clf__gamma': ['scale', 0.01]
    },
    {
        'selector': [RFECV(estimator=SVC(kernel="linear"), step=10, cv=3)],
        'clf__C': [0.1, 1, 10],
        'clf__gamma': ['scale', 0.01]
    }
]

print("--- Start SVM Feature Selection Experiment ---")

for i, (train_idx, val_idx) in enumerate(outer_cv.split(X_trainval, y_trainval)):
    X_ot, X_ov = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
    y_ot, y_ov = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

    pipe = Pipeline([
        ('selector', 'passthrough'),
        ('scaler', RobustScaler()),
        ('clf', SVC(probability=True, kernel='rbf', random_state=42))
    ])

    search = RandomizedSearchCV(pipe, param_grid, n_iter=15, cv=inner_cv, 
                                scoring='roc_auc', n_jobs=-1, random_state=42)
    search.fit(X_ot, y_ot)
    
    best_pipe = search.best_estimator_
    method_name = type(best_pipe.named_steps['selector']).__name__
    winning_methods.append(method_name)
    
    probs = best_pipe.predict_proba(X_ov)[:, 1]
    all_y_val.extend(y_ov)
    all_probs.extend(probs)
    
    print(f"Fold {i+1}: Beste methode = {method_name} (AUC: {roc_auc_score(y_ov, probs):.3f})")

# %% 5️⃣ RESULTATEN ANALYSE
final_auc = roc_auc_score(all_y_val, all_probs)
print(f"\nTotale Nested CV AUC: {final_auc:.4f}")
print("Verdeling winnende methodes:", pd.Series(winning_methods).value_counts())

# ROC Curve Plotten
fpr, tpr, _ = roc_curve(all_y_val, all_probs)
plt.plot(fpr, tpr, label=f'SVM (AUC = {final_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM Experiment')
plt.legend()
plt.show()