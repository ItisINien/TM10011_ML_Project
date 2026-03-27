# %% Univariate
from Forest_univariate import Forest_Uni

results = Forest_Uni()

nested_auc_forest_uni = results["nested_auc_uni"]
nested_f2_forest_uni = results["nested_f2_uni"]
aucs_forest_uni = results["fold_aucs_uni"]

print(f"Nested CV ROC-AUC Forest with univariate feature selection: {nested_auc_forest_uni:.3f}")
print(f"Nested CV F2-score Forest with univariate feature selection: {nested_f2_forest_uni:.3f}")
print(f"Aucs 5-folds Forest with univariate feature selection: {[round(float(x), 3) for x in aucs_forest_uni]}")

# %% Constant and Correlated features removed
from Forest_Cor_Const import Forest_const_cor

results = Forest_const_cor()

nested_auc_forest_const_cor = results["nested_auc_const_cor"]
nested_f2_forest_const_cor = results["nested_f2_const_cor"]
aucs_forest_const_cor = results["fold_aucs_const_cor"]

print(f"Nested CV ROC-AUC with Const and corr feature selection: {nested_auc_forest_const_cor:.3f}")
print(f"Nested CV F2-score with Const and corr feature selection: {nested_f2_forest_const_cor:.3f}")
print(f"Aucs 5-folds Forest with Const and corr feature selection: {[round(float(x), 3) for x in aucs_forest_const_cor]}")

# %% No feature Selection
from Forest_Only import Forest_Only_results

results = Forest_Only_results()

nested_auc_forest_only = results["nested_auc_only"]
nested_f2_forest_only = results["nested_f2_only"]
aucs_forest_only = results["fold_aucs_only"]

print(f"Nested CV ROC-AUC Forest no feature selection: {nested_auc_forest_only:.3f}")
print(f"Nested CV F2-score no feature selection: {nested_f2_forest_only:.3f}")
print(f"Aucs 5-folds Forest no feature selection: {[round(float(x), 3) for x in aucs_forest_only]}")

# %% optimization based feature selection werkt nog niet helemaal
# from Forest_optimization import Forest_opt

# results = Forest_opt()

# nested_auc_forest_opt = results["nested_auc_forest_opt"]
# nested_f2_forest_opt = results["nested_f2_forest_opt"]
# aucs_forest_opt = results["fold_aucs_forest_opt"]

# print(f"Nested CV ROC-AUC Forest optimization selection: {nested_auc_forest_opt:.3f}")
# print(f"Nested CV F2-score optimization selection: {nested_f2_forest_opt:.3f}")
# print(f"Aucs 5-folds Forest optimization feature selection: {[round(float(x), 3) for x in aucs_forest_opt]}")

# %%
from SVM_uni import SVM_Uni

results = SVM_Uni()

nested_auc_svm_uni = results["nested_auc_SVM_uni"]
nested_f2_svm_uni = results["nested_f2_SVM_uni"]
aucs_svm_uni = results["fold_aucs_SVM_uni"]

print(f"Nested CV ROC-AUC SVM univariate selection: {nested_auc_svm_uni:.3f}")
print(f"Nested CV F2-score SVM univariate selection: {nested_f2_svm_uni:.3f}")
print(f"Aucs 5-folds SVM univariate feature selection: {[round(float(x), 3) for x in aucs_svm_uni]}")

# %%
from LG import Logistic_Uni

results = Logistic_Uni()

nested_auc_lg_uni = results["nested_auc_lg"]
nested_f2_lg_uni = results["nested_f2_lg"]
aucs_lg_uni = results["fold_aucs_lg"]

print(f"Nested CV ROC-AUC LG univariate selection: {nested_auc_lg_uni:.3f}")
print(f"Nested CV F2-score LG univariate selection: {nested_f2_lg_uni:.3f}")
print(f"Aucs 5-folds LG univariate feature selection: {[round(float(x), 3) for x in aucs_lg_uni]}")

# %% BOXPLOT VAN 5-FOLD AUCs

import matplotlib.pyplot as plt

# Verzamel alle AUCs per methode in een dict
aucs_dict = {
    "RF Univariate": aucs_forest_uni,
    "RF Const+Corr": aucs_forest_const_cor,
    "RF No selection": aucs_forest_only,
    "SVM Univariate": aucs_svm_uni,
    "LG Univar + Optimisation": aucs_lg_uni,
    # Voeg hier eventueel nieuwe methodes toe of comment uit wat je niet wil plotten
    # "Nieuwe methode": aucs_nieuw
}

# Boxplot
plt.figure(figsize=(10,6))
plt.boxplot([aucs_dict[name] for name in aucs_dict], labels=[name for name in aucs_dict], patch_artist=True)

plt.ylabel("ROC-AUC per fold")
plt.title("5-fold ROC-AUC comparison per method")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
# %% SELECT BEST MODEL BASED ON NESTED AUC

nested_aucs = {
    "RF Const+Corr": nested_auc_forest_const_cor,
    "RF No selection": nested_auc_forest_only,
    "SVM Univariate": nested_auc_svm_uni,
    "LG Univariate": nested_auc_lg_uni
}

best_model_name = max(nested_aucs, key=nested_aucs.get)

print("\nBest model based on nested CV AUC:", best_model_name)
print("Nested AUC:", round(nested_aucs[best_model_name], 3))

# %%
# %% TESTSET EVALUATION

if best_model_name == "RF Const+Corr":
    from Forest_Cor_Const import Forest_const_cor
    results = Forest_const_cor(test=True)

elif best_model_name == "RF No selection":
    from Forest_Only import Forest_Only_results
    results = Forest_Only_results(test=True)

elif best_model_name == "SVM Univariate":
    from SVM_uni import SVM_Uni
    results = SVM_Uni(test=True)

elif best_model_name == "LG Univariate":
    from LG import Logistic_Uni
    results = Logistic_Uni(test=True)

print("\nTest set results:")
print("Test ROC-AUC:", round(results["test_auc"], 3))