# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!



#%% Data loading functions and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from worcliver.load_data import load_data
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import VarianceThreshold


from worcliver.load_data import load_data
data = load_data()



# %% Outliers bekijken
import numpy as np
numeric_cols = data.select_dtypes(include=[np.number])

# Bereken Z-scores
z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()

# Zoek outliers: rijen met minstens één Z-score > 3 of < -3
outliers = numeric_cols[(np.abs(z_scores) > 3).any(axis=1)]

print(f"Aantal rijen met outliers: {len(outliers)}")

outlier_values = (np.abs(z_scores) > 3).sum().sum()

print(f"Aantal outlier waarden: {outlier_values}")

total_values = numeric_cols.size

# Percentage
percentage_outliers = (outlier_values / total_values) * 100
print(f"Procent outlier waarden: {percentage_outliers}")


# %% Outlier bekijken

# Selecteer numerieke kolommen
numeric_cols = data.select_dtypes(include=[np.number])

# Bereken Z-scores
z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()

# Boolean mask: True als |Z| > 3
outlier_mask = np.abs(z_scores) > 3

# Aantal outliers per feature
outliers_per_feature = outlier_mask.sum()

# Percentage outliers per feature
percentage_outliers_per_feature = (outliers_per_feature / len(numeric_cols)) * 100

# Sorteer features op percentage outliers (hoogste eerst)
percentage_outliers_per_feature = percentage_outliers_per_feature.sort_values(ascending=False)

# Toon features met hun percentage outliers
print(percentage_outliers_per_feature)

# %% Schaler
from sklearn.preprocessing import RobustScaler
import pandas as pd

# Selecteer numerieke kolommen
numeric_cols = data.select_dtypes(include=[np.number])

# Initialiseer de RobustScaler
scaler = RobustScaler()

# Pas scaling toe
scaled_numeric = scaler.fit_transform(numeric_cols)

# Zet het terug naar een DataFrame (optioneel)
scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_cols.columns, index=numeric_cols.index)

non_numeric = data.select_dtypes(exclude=[np.number])
final_data = pd.concat([scaled_numeric_df, non_numeric], axis=1)

# %% met outliers gescaled
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd

# 1) Targetvariabele
y = final_data['label'].map({'benign':0, 'malignant':1})

# 2) Feature matrix: drop label
x = final_data.drop(columns=['label'])

# 3) Split features per klasse
x_benign = x[y == 0]
x_malign = x[y == 1]

# 4) T-test per feature
t_values = []
p_values = []

for col in x.columns:
    t, p = ttest_ind(
        x_benign[col].values,
        x_malign[col].values,
        equal_var=False,    # Welch's t-test
        nan_policy='omit'   # negeer NaN
    )
    t_values.append(t)
    p_values.append(p)

# 5) Resultaten in DataFrame
results = pd.DataFrame({
    'feature': x.columns,
    't_statistics': t_values,
    'p_value': p_values,
    'abs_t': np.abs(t_values)
}).sort_values(['p_value', 'abs_t'], ascending=[True, False])

# 6) Top 50 features
top50 = results.head(50)
top50_features = top50['feature'].tolist()

# Print
print(top50[["feature", "t_statistics", "p_value"]].to_string(index=False))
 # %% Zonder outliers gescaled

 from scipy.stats import ttest_ind
import pandas as pd
import numpy as np

# Targetvariabele
y_orig = data['label'].map({'benign':0, 'malignant':1})

# Feature matrix (zonder label)
x_orig = data.drop(columns=['label'])

# Split per klasse
x_benign_orig = x_orig[y_orig == 0]
x_malign_orig = x_orig[y_orig == 1]

# T-test per feature
t_values_orig = []
p_values_orig = []

for col in x_orig.columns:
    t, p = ttest_ind(
        x_benign_orig[col].values,
        x_malign_orig[col].values,
        equal_var=False,
        nan_policy='omit'
    )
    t_values_orig.append(t)
    p_values_orig.append(p)

# Resultaten
results_orig = pd.DataFrame({
    'feature': x_orig.columns,
    't_statistics': t_values_orig,
    'p_value': p_values_orig,
    'abs_t': np.abs(t_values_orig)
}).sort_values(['p_value', 'abs_t'], ascending=[True, False])

# Top 50 features
top50_orig = results_orig.head(50)
top50_features_orig = top50_orig['feature'].tolist()

print("Top 50 features (originele data met outliers):")
print(top50_orig[["feature", "t_statistics", "p_value"]].to_string(index=False))

top50_features_scaled = top50_features

# %% vergelijken
only_orig = set(top50_features_orig) - set(top50_features_scaled)
print("Features alleen in originele data:", only_orig)

only_scaled = set(top50_features_scaled) - set(top50_features_orig)
print("Features alleen in geschaalde data:", only_scaled)

# %% boxplots
import matplotlib.pyplot as plt
import seaborn as sns

# De vier features
features_to_plot = [
    'PREDICT_original_tf_LBP_peak_R3_P12',
    'PREDICT_original_tf_LBP_peak_R15_P36',
    'PREDICT_original_logf_mean_sigma1',
    'PREDICT_original_tf_Gabor_quartile_range_F0.05_A1.57'
]

# Target variabele
y = data['label']

# Maak losse boxplots
for feat in features_to_plot:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=y, y=data[feat])
    plt.title(f'Boxplot van {feat} per klasse')
    plt.xlabel('Klasse')
    plt.ylabel('Waarde')
    plt.show()

# %% 2 beste features

# %%

import matplotlib.pyplot as plt
import seaborn as sns

# Kies de twee beste features (top2 uit je t-test selectie)
best_features = top50_features_scaled[:2]  # eerste twee features

plt.figure(figsize=(6,6))
sns.scatterplot(
    x=final_data[best_features[0]],
    y=final_data[best_features[1]],
    hue=final_data['label'],
    palette={'benign':'blue', 'malignant':'red'},
    alpha=0.7
)

plt.xlabel(best_features[0])
plt.ylabel(best_features[1])
plt.title(f'Scatterplot van de twee beste features')
plt.legend(title='Klasse')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Stabiele features

# --- STAP 1: Dataset laden ---
data = load_data()
X = data.select_dtypes(include=[np.number])
y = data['label']


# --- STAP 2: K-Fold voorbereiden ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# We maken een lijst om de top-features per fold in op te slaan
features_per_fold_10= []
features_per_fold_20= []
features_per_fold_30= []
features_per_fold_40= []
features_per_fold_50= []
features_per_fold_60= []
features_per_fold_70=[]


print("Stap 1: Zoeken naar stabiele features in alle folds...")

# --- EERSTE LOOP: FEATURES ZOEKEN ---
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    
    # Schalen (alleen op train)
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    
    # Significantie check (p < 0.05)
    f_vals, p_vals = f_classif(X_train_scaled, y_train)
    sig_indices = np.where(p_vals < 0.05)[0]
    
    if len(sig_indices) > 0:
        # Pak de top 20 van deze fold
        sorted_sig_indices = sig_indices[np.argsort(f_vals[sig_indices])[::-1]]

        top_indices = sorted_sig_indices[:10]
        features_per_fold_10.append(set(X.columns[top_indices]))

        top_indices = sorted_sig_indices[:20]
        features_per_fold_20.append(set(X.columns[top_indices]))

        top_indices = sorted_sig_indices[:30]
        features_per_fold_30.append(set(X.columns[top_indices]))

        top_indices = sorted_sig_indices[:40]
        features_per_fold_40.append(set(X.columns[top_indices]))

        top_indices = sorted_sig_indices[:50]
        features_per_fold_50.append(set(X.columns[top_indices]))

        top_indices = sorted_sig_indices[:60]
        features_per_fold_60.append(set(X.columns[top_indices]))

        top_indices = sorted_sig_indices[:70]
        features_per_fold_70.append(set(X.columns[top_indices]))


# Vind de doorsnede: welke features staan in ELKE fold in de top 50?
stabiele_features_10 = list(set.intersection(*features_per_fold_10))
stabiele_features_20 = list(set.intersection(*features_per_fold_20))
stabiele_features_30 = list(set.intersection(*features_per_fold_30))
stabiele_features_40 = list(set.intersection(*features_per_fold_40))
stabiele_features_50 = list(set.intersection(*features_per_fold_50))
stabiele_features_60 = list(set.intersection(*features_per_fold_60))
stabiele_features_70 = list(set.intersection(*features_per_fold_70))


print(f"\nAantal stabiele features gevonden (in alle 5 folds) top 10: {len(stabiele_features_10)}")
print(f"Features: {stabiele_features_10}\n")

print(f"\nAantal stabiele features gevonden (in alle 5 folds) top 20: {len(stabiele_features_20)}")
print(f"Features: {stabiele_features_20}\n")

print(f"\nAantal stabiele features gevonden (in alle 5 folds) top 30: {len(stabiele_features_30)}")
print(f"Features: {stabiele_features_30}\n")

print(f"\nAantal stabiele features gevonden (in alle 5 folds) top 40: {len(stabiele_features_40)}")
print(f"Features: {stabiele_features_40}\n")

print(f"\nAantal stabiele features gevonden (in alle 5 folds) top 50: {len(stabiele_features_50)}")
print(f"Features: {stabiele_features_50}\n")


print(f"\nAantal stabiele features gevonden (in alle 5 folds) top 60: {len(stabiele_features_60)}")
print(f"Features: {stabiele_features_60}\n")

print(f"\nAantal stabiele features gevonden (in alle 5 folds) top 70: {len(stabiele_features_70)}")
print(f"Features: {stabiele_features_70}\n")
  
# %% trainen voor stabiele features
stabiele_features_dict = {
    10: stabiele_features_10,
    20: stabiele_features_20,
    30: stabiele_features_30,
    40: stabiele_features_40,
    50: stabiele_features_50,
    60: stabiele_features_60,
    70: stabiele_features_70
}
top_N_list = []
weighted_f1_list = []

# Loop over alle top-N stabiele features
for n, stabiele_features in stabiele_features_dict.items():

    if len(stabiele_features) == 0:
        print(f"FOUT: Geen stabiele features gevonden voor top {n}")
        continue  # ga naar de volgende top-N
    
    all_y_test = []
    all_y_pred = []

    print(f"\nStap 2: Model trainen met top {n} stabiele features...")

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Filter alleen stabiele features
        X_train_stabiel = X_train[stabiele_features]
        X_test_stabiel = X_test[stabiele_features]
        
        # Schalen
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_stabiel)
        X_test_scaled = scaler.transform(X_test_stabiel)
        
        # Random Forest trainen
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        # Voorspellen
        y_pred = clf.predict(X_test_scaled)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
    
    # --- Resultaten ---
    # print("\n" + "="*40)
    # print(f"RESULTATEN MET TOP {n} STABIELE FEATURES")
    # print("="*40)
    # print(classification_report(all_y_test, all_y_pred))
    
    # # Confusion matrix
    # plt.figure(figsize=(8, 6))
    # cm = confusion_matrix(all_y_test, all_y_pred)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
    #             xticklabels=clf.classes_, yticklabels=clf.classes_)
    # plt.title(f'Confusion Matrix (top {n} stabiele features)')
    # plt.ylabel('Werkelijke Diagnose')
    # plt.xlabel('Voorspelling')
    # plt.show()
    
    print(f"Aantal features gebruikt: {len(stabiele_features)}\n")

    report = classification_report(all_y_test, all_y_pred, output_dict=True)
    weighted_f1 = report['weighted avg']['f1-score']
    top_N_list.append(n)
    weighted_f1_list.append(weighted_f1)


# %% grafiek features
aantal_features=[len(stabiele_features_10),len(stabiele_features_20), 
                 len(stabiele_features_30), len(stabiele_features_40),len(stabiele_features_50),
                   len(stabiele_features_60), len(stabiele_features_70) ]
print(aantal_features)
print(weighted_f1_list)

plt.figure(figsize=(8,6))
plt.plot(aantal_features, weighted_f1_list, marker='o')
plt.xlabel("Aantal stabiele features")
plt.ylabel("Weighted F1-score")
plt.title("Weighted F1-score vs aantal stabiele features")
plt.grid(True)
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from worcliver.load_data import load_data
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC  # Nieuwe import: Support Vector Classifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- STAP 1: Dataset laden ---
data = load_data()
X = data.select_dtypes(include=[np.number])
y = data['label']

# --- STAP 2: K-Fold voorbereiden ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
features_per_fold = []

print("Stap 1: Zoeken naar stabiele features in alle folds...")

# --- EERSTE LOOP: FEATURES ZOEKEN (Dezelfde methode als net) ---
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    
    f_vals, p_vals = f_classif(X_train_scaled, y_train)
    sig_indices = np.where(p_vals < 0.05)[0]
    
    if len(sig_indices) > 0:
        sorted_sig_indices = sig_indices[np.argsort(f_vals[sig_indices])[::-1]]
        top_indices = sorted_sig_indices[:150]
        features_per_fold.append(set(X.columns[top_indices]))

stabiele_features = list(set.intersection(*features_per_fold))
print(f"\nAantal stabiele features gevonden: {len(stabiele_features)}")

if len(stabiele_features) == 0:
    print("Geen stabiele features gevonden. Probeer de selectie te verruimen.")
else:
    # --- TWEEDE LOOP: TRAINING MET SVM ---
    all_y_test = []
    all_y_pred = []

    print("Stap 2: SVM model trainen op stabiele features...")
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train_stabiel = X_train[stabiele_features]
        X_test_stabiel = X_test[stabiele_features]
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_stabiel)
        X_test_scaled = scaler.transform(X_test_stabiel)
        
        # --- HIER ZIT DE VERANDERING: SVM ---
        # kernel='linear' is vaak het best voor medische interpretatie
        # probability=True zorgt dat we later ook kansberekeningen kunnen doen
        clf = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    # --- FINALE RESULTATEN ---
    # print("\n" + "="*40)
    # print("RESULTATEN MET SVM & STABIELE FEATURES")
    # print("="*40)
    # print(classification_report(all_y_test, all_y_pred))

    # # Confusion Matrix
    # plt.figure(figsize=(8, 6))
    # cm = confusion_matrix(all_y_test, all_y_pred)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
    #             xticklabels=clf.classes_, yticklabels=clf.classes_)
    # plt.title(f'SVM Confusion Matrix ({len(stabiele_features)} Features)')
    # plt.ylabel('Werkelijke Diagnose')
    # plt.xlabel('Voorspelling')
    # plt.show()

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# --- Instellingen ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

performance_list = []

feature_steps = list(range(1, len(stabiele_features)+1, 1))
if feature_steps[-1] != len(stabiele_features):
    feature_steps.append(len(stabiele_features))  # zorg dat laatste stap exact alle features is


# --- Loop over aantal features van 1 tot 42 ---
for n_features in feature_steps:
    top_n_features = stabiele_features[:n_features]
    
    all_y_test = []
    all_y_pred = []

    for train_index, test_index in skf.split(X, y):
        X_train = X.iloc[train_index][top_n_features]
        X_test  = X.iloc[test_index][top_n_features]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        clf = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    report = classification_report(all_y_test, all_y_pred, output_dict=True)
    performance_list.append(report['weighted avg']['f1-score'])

# --- Plot performance vs aantal features ---
plt.figure(figsize=(8,6))
plt.plot(feature_steps, performance_list, marker='o')
plt.xlabel("Aantal features gebruikt")
plt.ylabel("Weighted F1-score")
plt.title("F1-score SVM vs aantal stabiele features")
plt.grid(True)
plt.show()

