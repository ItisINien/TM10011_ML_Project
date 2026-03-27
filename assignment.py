# %% IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

from worcliver.load_data import load_data


# %% 1️⃣ CUSTOM TRANSFORMER
class CorrAndSelect(BaseEstimator, TransformerMixin):
    def __init__(self, k=200, corr_threshold=0.9):
        self.k = k
        self.corr_threshold = corr_threshold
    
    def fit(self, X, y):
        corr_matrix = X.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        
        X_filtered = X.drop(columns=self.to_drop_)
        
        self.selector_ = SelectKBest(score_func=f_classif,
                                     k=min(self.k, X_filtered.shape[1]))
        self.selector_.fit(X_filtered, y)
        
        self.features_ = X_filtered.columns[self.selector_.get_support()]
        return self
    
    def transform(self, X):
        X_filtered = X.drop(columns=self.to_drop_)
        return pd.DataFrame(
            self.selector_.transform(X_filtered),
            columns=self.features_,
            index=X.index
        )


# %% 2️⃣ LOAD DATA
data = load_data()
X = data.select_dtypes(include=[np.number])
y = data['label'].map({'benign': 0, 'malignant': 1})


# %% 3️⃣ TRAIN / TEST SPLIT
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# %% 4️⃣ NESTED CROSS-VALIDATION
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

all_y_outer = []
all_y_outer_proba = []
outer_fold_aucs = []
coef_list = []

param_dist = {
    'feat_select__k': [150, 200, 250],
    'rfe__n_features_to_select': [8, 10, 12],
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs']
}

for outer_train_idx, outer_val_idx in outer_cv.split(X_trainval, y_trainval):

    X_outer_train = X_trainval.iloc[outer_train_idx].copy()
    X_outer_val = X_trainval.iloc[outer_val_idx].copy()
    y_outer_train = y_trainval.iloc[outer_train_idx]
    y_outer_val = y_trainval.iloc[outer_val_idx]

    # 🔹 Pipeline
    pipeline = Pipeline([
        ('feat_select', CorrAndSelect(k=200, corr_threshold=0.9)),
        ('rfe', RFE(
            estimator=LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                max_iter=1000
            ),
            n_features_to_select=10,
            step=0.1
        )),
        ('scaler', RobustScaler()),
        ('clf', LogisticRegression(
            max_iter=1000,
            penalty='l2',
            solver='lbfgs'
        ))
    ])

    # 🔹 Inner CV hyperparameter tuning
    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=30,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )

    grid_search.fit(X_outer_train, y_outer_train)

    print(f"Best params (inner CV): {grid_search.best_params_}")
    print(f"Best inner CV ROC-AUC: {grid_search.best_score_:.3f}")

    # 🔹 Evaluate on outer fold
    best_model = grid_search.best_estimator_
    y_outer_proba = best_model.predict_proba(X_outer_val)[:, 1]

    fold_auc = roc_auc_score(y_outer_val, y_outer_proba)
    outer_fold_aucs.append(fold_auc)

    print(f"Outer fold ROC-AUC: {fold_auc:.3f}\n")

    all_y_outer.extend(y_outer_val)
    all_y_outer_proba.extend(y_outer_proba)

    # 🔹 Extract final selected features after RFE
    features_after_uni = best_model.named_steps['feat_select'].features_
    rfe_mask = best_model.named_steps['rfe'].support_
    final_features = features_after_uni[rfe_mask]

    coefs = best_model.named_steps['clf'].coef_.ravel()

    coef_list.append(pd.DataFrame({
        'feature': final_features,
        'coefficient': coefs
    }))


# %% 5️⃣ NESTED CV PERFORMANCE
nested_auc = roc_auc_score(all_y_outer, all_y_outer_proba)

print("Nested CV ROC-AUC:", round(nested_auc, 3))
print("Outer fold mean AUC:", round(np.mean(outer_fold_aucs), 3))
print("Outer fold std AUC:", round(np.std(outer_fold_aucs), 3))
print("Outer fold range:",
      round(min(outer_fold_aucs), 3),
      "-",
      round(max(outer_fold_aucs), 3))


# %% 6️⃣ COEFFICIENT ANALYSIS
coef_df = pd.concat(coef_list)
coef_mean = coef_df.groupby('feature')['coefficient'].mean()

coef_mean_sorted = coef_mean.reindex(
    coef_mean.abs().sort_values(ascending=False).index
)

print("\nTop 10 features:")
print(coef_mean_sorted.head(10))

plt.figure(figsize=(8,6))
plt.barh(coef_mean_sorted.index[:10][::-1],
         coef_mean_sorted.values[:10][::-1])
plt.xlabel("Mean coefficient")
plt.title("Top 10 Logistic Regression Features")
plt.show()


# %% 7️⃣ FINAL MODEL ON TEST SET
# Gebruik hier de meest gekozen hyperparameters uit nested CV
pipeline_final = Pipeline([
    ('feat_select', CorrAndSelect(k=150, corr_threshold=0.9)),
    ('rfe', RFE(
        estimator=LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=1000
        ),
        n_features_to_select=10
    )),
    ('scaler', RobustScaler()),
    ('clf', LogisticRegression(
        C=10,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000
    ))
])

pipeline_final.fit(X_trainval, y_trainval)

y_test_proba = pipeline_final.predict_proba(X_test)[:,1]
test_auc = roc_auc_score(y_test, y_test_proba)

print("\nFinal Test ROC-AUC:", round(test_auc, 3))



# # %% IMPORTS
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import RobustScaler
# from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import RFE, SelectKBest, f_classif
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import roc_auc_score
# from sklearn.base import BaseEstimator, TransformerMixin
# from worcliver.load_data import load_data

# # %% 1️⃣ CUSTOM TRANSFORMER (zelfde als jij)
# class CorrAndSelect(BaseEstimator, TransformerMixin):
#     def __init__(self, k=200, corr_threshold=0.9):
#         self.k = k
#         self.corr_threshold = corr_threshold
    
#     def fit(self, X, y):
#         corr_matrix = X.corr(method='spearman').abs()
#         upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#         self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
#         X_filtered = X.drop(columns=self.to_drop_)
        
#         self.selector_ = SelectKBest(score_func=f_classif, k=min(self.k, X_filtered.shape[1]))
#         self.selector_.fit(X_filtered, y)
        
#         self.features_ = X_filtered.columns[self.selector_.get_support()]
#         return self
    
#     def transform(self, X):
#         X_filtered = X.drop(columns=self.to_drop_)
#         return pd.DataFrame(self.selector_.transform(X_filtered), columns=self.features_)

# # %% 2️⃣ LOAD DATA
# data = load_data()
# X = data.select_dtypes(include=[np.number])
# y = data['label'].map({'benign': 0, 'malignant': 1})

# # %% 3️⃣ TRAIN/TEST SPLIT
# X_trainval, X_test, y_trainval, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # %% 4️⃣ NESTED CV
# outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# all_y_outer = []
# all_y_outer_proba = []

# coef_list = []

# # Hyperparameter grid voor Logistic Regression
# param_dist = {
#     'feat_select__k': [150, 200, 250],
#     'rfe__n_features_to_select': [8, 10, 12],
#     'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'clf__penalty': ['l2'],   # simpel houden, dus geen penalty. Kan net zo goed weg. 
#     'clf__solver': ['lbfgs']  # compatible met l2.
# }

# for outer_train_idx, outer_val_idx in outer_cv.split(X_trainval, y_trainval):
#     X_outer_train = X_trainval.iloc[outer_train_idx].copy()
#     X_outer_val = X_trainval.iloc[outer_val_idx].copy()
#     y_outer_train = y_trainval.iloc[outer_train_idx]
#     y_outer_val = y_trainval.iloc[outer_val_idx]

#     # pipeline = Pipeline([
#     #     ('feat_select', CorrAndSelect(k=200, corr_threshold=0.9)),
#     #     ('scaler', RobustScaler()),  # BELANGRIJK voor LR
#     #     ('clf', LogisticRegression(max_iter=1000))
#     # ])

# # Pipeline: eerst univariate, dan RFE, dan schalen, dan LR. RFE is belangrijk omdat het interacties tussen features kan meenemen, wat univariate selectie niet doet.
# pipeline = Pipeline([
#     ('feat_select', CorrAndSelect(k=200, corr_threshold=0.9)),  # eerst correlatie + univariate
#     ('rfe', RFE(
#         estimator=LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000),
#         n_features_to_select=50,
#         step=0.1
#     )),
#     ('scaler', RobustScaler()),
#     ('clf', LogisticRegression(max_iter=1000, penalty='l2', solver='lbfgs'))
# ])

# #%%
#     # RandomizedSearchCV (inner loop)
#     grid_search = RandomizedSearchCV(
#         estimator=pipeline,
#         param_distributions=param_dist,
#         n_iter=30,
#         cv=inner_cv,
#         scoring='roc_auc',
#         n_jobs=-1,
#         random_state=42
#     )

#     grid_search.fit(X_outer_train, y_outer_train)

#     print(f"Outer fold best params: {grid_search.best_params_}")
#     print(f"Outer fold best inner CV ROC-AUC: {grid_search.best_score_:.3f}")

#     # Evaluatie op outer fold
#     best_model = grid_search.best_estimator_
#     y_outer_proba = best_model.predict_proba(X_outer_val)[:,1]

#     all_y_outer.extend(y_outer_val)
#     all_y_outer_proba.extend(y_outer_proba)

#     fold_auc = roc_auc_score(y_outer_val, y_outer_proba)
#     print(f"Outer fold AUC: {fold_auc:.3f}")

# print("5-fold CV ROC-AUC mean:", cv_scores.mean())
# print("5-fold CV ROC-AUC std:", cv_scores.std())

# # features na RFE matchen met coëfficiënten
# features_after_uni = best_model.named_steps['feat_select'].features_
# rfe_mask = best_model.named_steps['rfe'].support_
# final_features = features_after_uni[rfe_mask]

# coefs = best_model.named_steps['clf'].coef_.ravel()  # 1D array

# assert len(final_features) == len(coefs), "Mismatch features vs coefs!"

# coef_list.append(pd.DataFrame({
#     'feature': final_features,
#     'coefficient': coefs
# }))

# # %% 5️⃣ EVALUATE NESTED CV
# roc_auc = roc_auc_score(all_y_outer, all_y_outer_proba)
# print(f"Nested CV ROC-AUC: {roc_auc:.3f}")

# # %% 6️⃣ COEFFICIENT ANALYSE
# coef_df = pd.concat(coef_list)
# coef_mean = coef_df.groupby('feature')['coefficient'].mean()

# # Sorteer op absolute waarde (belangrijk!)
# coef_mean_sorted = coef_mean.reindex(coef_mean.abs().sort_values(ascending=False).index)

# print("Top features (LR):")
# print(coef_mean_sorted.head(20))

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,6))
# plt.barh(coef_mean_sorted.index[:20][::-1], coef_mean_sorted.values[:20][::-1])
# plt.xlabel('Mean coefficient')
# plt.title('Top 20 features Logistic Regression')
# plt.show()

# # %% (OPTIONEEL) FINAL MODEL OP TESTSET
# # pipeline_final = Pipeline([
# #     ('feat_select', CorrAndSelect(k=200, corr_threshold=0.9)),
# #     ('scaler', RobustScaler()),
# #     ('clf', LogisticRegression(
# #         C=1,
# #         penalty='l2',
# #         solver='lbfgs',
# #         max_iter=1000
# #     ))
# # ])
# #
# # pipeline_final.fit(X_trainval, y_trainval)
# # y_test_proba = pipeline_final.predict_proba(X_test)[:,1]
# # test_auc = roc_auc_score(y_test, y_test_proba)
# # print(f"Test ROC-AUC: {test_auc:.3f}")
# # %%

# %%
