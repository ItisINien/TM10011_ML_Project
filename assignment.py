# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!


#%% Data loading functions. Uncomment the one you want to use
#from worcgist.load_data import load_data
#from worclipo.load_data import load_data
from worcliver.load_data import load_data
#from hn.load_data import load_data
#from ecg.load_data import load_data
data = load_data()


# %%
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


# %%

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