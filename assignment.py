# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!

#%% Data loading functions and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from worcliver.load_data import load_data
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from worcliver.load_data import load_data
from collections import Counter
from sklearn.metrics import roc_auc_score, accuracy_score

# %% LOAD DATA
data = load_data()
X = data.select_dtypes(include=[np.number])
y = data['label'].map({'benign': 0, 'malignant': 1})

# %% TRAIN/TEST SPLIT (test set apart, maar we gebruiken hem hier niet)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %% K-FOLD SETUP
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# %% STEP 1: IDENTIFY STABLE FEATURES
features_per_fold = []

for train_idx, val_idx in cv.split(X_train, y_train):
    X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
    
    scaler = RobustScaler()
    X_fold_scaled = pd.DataFrame(scaler.fit_transform(X_fold_train), columns=X_train.columns)
    
    f_vals, p_vals = f_classif(X_fold_scaled, y_fold_train)
    sig_indices = np.where(p_vals < 0.05)[0]
    
    if len(sig_indices) > 0:
        sorted_sig_indices = sig_indices[np.argsort(f_vals[sig_indices])[::-1]]
        top_indices = sorted_sig_indices[:30]
        features_per_fold.append(list(X_train.columns[top_indices]))

all_features = [f for fold in features_per_fold for f in fold]
counter = Counter(all_features)
stabiele_features_14 = [feature for feature, _ in counter.most_common(14)]
print("Top 14 stabiele features over alle folds:")
print(stabiele_features_14)

# %% STEP 2: ROC-AUC AND ACCURACY ON VALIDATION FOLDS
all_y_val = []
all_y_val_proba = []

for train_idx, val_idx in cv.split(X_train, y_train):
    X_fold_train = X_train.iloc[train_idx][stabiele_features_14]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx][stabiele_features_14]
    y_fold_val = y_train.iloc[val_idx]
    
    scaler = RobustScaler()
    X_fold_train_scaled = scaler.fit_transform(X_fold_train)
    X_fold_val_scaled = scaler.transform(X_fold_val)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_fold_train_scaled, y_fold_train)
    
    y_val_proba = clf.predict_proba(X_fold_val_scaled)[:,1]
    
    all_y_val.extend(y_fold_val)
    all_y_val_proba.extend(y_val_proba)

# ROC-AUC
roc_auc = roc_auc_score(all_y_val, all_y_val_proba)

# Accuracy (threshold 0.5)
y_val_pred_labels = [1 if p >= 0.5 else 0 for p in all_y_val_proba]
accuracy = accuracy_score(all_y_val, y_val_pred_labels)

print(f"\nROC-AUC over alle validation folds: {roc_auc:.3f}")
print(f"Accuracy (threshold 0.5) over alle validation folds: {accuracy:.3f}")

 # %%
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import RobustScaler
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report, accuracy_score
# import matplotlib.pyplot as plt

# # --- Instellingen ---
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# performance_list = []

# feature_steps = list(range(1, len(stabiele_features)+1, 1))
# if feature_steps[-1] != len(stabiele_features):
#     feature_steps.append(len(stabiele_features))  # zorg dat laatste stap exact alle features is


# # --- Loop over aantal features van 1 tot 42 ---
# for n_features in feature_steps:
#     top_n_features = stabiele_features[:n_features]
    
#     all_y_test = []
#     all_y_pred = []

#     for train_index, test_index in skf.split(X, y):
#         X_train = X.iloc[train_index][top_n_features]
#         X_test  = X.iloc[test_index][top_n_features]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#         scaler = RobustScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled  = scaler.transform(X_test)

#         clf = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
#         clf.fit(X_train_scaled, y_train)
#         y_pred = clf.predict(X_test_scaled)

#         all_y_test.extend(y_test)
#         all_y_pred.extend(y_pred)

#     report = classification_report(all_y_test, all_y_pred, output_dict=True)
#     performance_list.append(report['weighted avg']['f1-score'])

# # --- Plot performance vs aantal features ---
# plt.figure(figsize=(8,6))
# plt.plot(feature_steps, performance_list, marker='o')
# plt.xlabel("Aantal features gebruikt")
# plt.ylabel("Weighted F1-score")
# plt.title("F1-score SVM vs aantal stabiele features")
# plt.grid(True)
# plt.show()

