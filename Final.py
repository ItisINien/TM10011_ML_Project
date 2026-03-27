# %% Univariate
from Forest_univariate import Forest_Uni

results = Forest_Uni()

nested_auc_forest_uni = results["nested_auc_uni"]
nested_f2_forest_uni = results["nested_f2_uni"]
aucs_forest_uni = results["fold_aucs"]

print(f"Nested CV ROC-AUC Forest with univariate feature selection: {nested_auc_forest_uni:.3f}")
print(f"Nested CV F2-score Forest with univariate feature selection: {nested_f2_forest_uni:.3f}")
print(f"Aucs 5-folds Forest with univariate feature selection: {aucs_forest_uni:.3f}")
# %% Constant and Correlation
from Forest_Cor_Const import Forest_const_cor

results = Forest_const_cor()

nested_auc_forest_const_cor = results["nested_auc_const_cor"]
nested_f2_forest_const_cor = results["nested_f2_const_cor"]
aucs_forest_const_cor = results["fold_aucs_const_cor"]

print(f"Nested CV ROC-AUC with Const and corr feature selection: {nested_auc_forest_const_cor:.3f}")
print(f"Nested CV F2-score with Const and corr feature selection: {nested_f2_forest_const_cor:.3f}")
print(f"Aucs 5-folds Forest with Const and corr feature selection: {aucs_forest_const_cor:.3f}")

# %% No feature Selection
from Forest_Only import Forest_Only_results

results = Forest_Only_results()

nested_auc_forest_only = results["nested_auc_only"]
nested_f2_forest_only = results["nested_f2_only"]

print(f"Nested CV ROC-AUC Forest no feature selection: {nested_auc_forest_only:.3f}")
print(f"Nested CV F2-score no feature selection: {nested_f2_forest_only:.3f}")

