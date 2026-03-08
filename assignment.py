


# proberen
from worcliver.load_data import load_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

data = load_data() # Laad de dataset

# %% Outlier detectie met Z-scores
numeric_cols = data.select_dtypes(include=[np.number])
z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std() # Bereken Z-scores om outliers te identificeren
outliers = numeric_cols[(np.abs(z_scores) > 3).any(axis=1)] # Zoek outliers: rijen met minstens één Z-score > 3 of < -3
print(f"Aantal rijen met outliers: {len(outliers)}")
outlier_values = (np.abs(z_scores) > 3).sum().sum()
print(f"Aantal outlier waarden: {outlier_values}")
total_values = numeric_cols.size
percentage_outliers = (outlier_values / total_values) * 100 # Percentage
print(f"Procent outlier waarden: {percentage_outliers}")

# %% Schalen van de numerieke kolommen met RobustScaler
numeric_cols = data.select_dtypes(include=[np.number]) # Selecteer alleen de numerieke kolommen 
scaler = RobustScaler() # Initialiseer de RobustScaler
scaled_values = scaler.fit_transform(numeric_cols) # Pas de scaler toe op de numerieke kolommen
data_scaled = pd.DataFrame(scaled_values, columns=numeric_cols.columns, index=numeric_cols.index) # Maak een nieuwe DataFrame met de geschaalde waarden en behoud de originele index en kolomnamen
print(data_scaled.head())

# Scalen check; Visualisatie van het effect van RobustScaler op één feature
import matplotlib.pyplot as plt
import seaborn as sns

feature_naam = numeric_cols.columns[0] # kiezen één feature om het effect te laten zien, kunt hier elke kolomnaam invullen
fig, axes = plt.subplots(1, 2, figsize=(15, 6)) # Maak een figuur met twee subplots (1 rij, 2 kolommen)

sns.boxplot(data=numeric_cols[feature_naam], ax=axes[0], color='skyblue') # 1. Boxplot VOOR schalen
axes[0].set_title(f'VOOR RobustScaling\n({feature_naam})')
axes[0].set_ylabel('Originele Waarde')

sns.boxplot(data=data_scaled[feature_naam], ax=axes[1], color='lightgreen') # 2. Boxplot NA schalen
axes[1].set_title(f'NA RobustScaling\n({feature_naam})')
axes[1].set_ylabel('Geschaalde Waarde (Mediaan = 0)')

plt.tight_layout()
plt.show()

# %% Stratified K-Fold
from sklearn.model_selection import StratifiedKFold
X = data_scaled # Gebruik de geschaalde data als features
y = data['label'] # Colom van benigne/maligne selecteren 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # n_splits=5 -> 5 groepen, shuffle=True -> willekeurige verdeling, random_state=42 -> reproduceerbaar
print(skf)

# K-fold check; verdeling controleren per fold

print("Totale dataset verdeling:") # Bekijk eerst de verdeling in je totale dataset (y)
print(y.value_counts(normalize=True) * 100)
print("-" * 30)

for i, (train_index, test_index) in enumerate(skf.split(X, y)): # 2. Loop door de 5 folds om de test-set van elke fold te checken
    y_test_fold = y.iloc[test_index]    # Pak de labels van de patiënten in de huidige test-fold
    percentages = y_test_fold.value_counts(normalize=True) * 100    # Bereken de percentages (bijv. hoeveel % is maligne)
    print(f"Fold {i+1}:")
    print(f"  Aantal patiënten: {len(y_test_fold)}")
    print(f"  Verdeling per klasse (%):")
    print(percentages)
    print("-" * 30)


# %% Feature selectie
from sklearn.feature_selection import SelectKBest, f_classif

scores = []# We maken een lijstje om de resultaten per fold in op te slaan
features_per_fold = [] # We maken een lijstje om de gekozen features per fold in op te slaan, zodat we later kunnen vergelijken welke features in meerdere folds voorkomen

for i, (train_index, test_index) in enumerate(skf.split(X, y)):# Hier begint de loop door de 5 folds

    X_train, X_test = X.iloc[train_index], X.iloc[test_index] # data slipsten dus train: 4 folds en Test: 1 fold
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    selector = SelectKBest(score_func=f_classif, k=10) # Feature selectie met de ANOVA F-test, kies de top 10 features
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    X_test_selected = selector.transform(X_test)   # Pas dezelfde selectie (dezelfde kolommen) toe op de testset

    cols = selector.get_support(indices=True)   # Controle Welke features zijn gekozen in deze fold?
    features_in_fold = X.columns[cols] # Pak de namen van de gekozen features in deze fold
    features_per_fold.append(set(features_in_fold)) # Sla de gekozen features van deze fold op in een lijst (set) om later te vergelijken tussen folds

    
    print(f"FOLD {i+1}:") # print welke fold we aan het trainen zijn
    print(f"  Training op {len(X_train)} patiënten, testen op {len(X_test)}") # print aantal patiënten in train en test van deze fold
    print(f"  Top feature in deze fold: {features_in_fold[0]}") # print de beste feature van deze fold, vrij nutteloos maar prima
    print("-" * 30) 

 
    p_waarden_alle = selector.pvalues_ # p-waarden check; Bekijk de p-waarden van de geselecteerde features per fold
    p_waarden_top10 = p_waarden_alle[cols] # Pak alleen de p-waarden van de 10 gekozen features
    
    print(f"--- SIGNIFICANTIE CHECK FOLD {i+1} ---") # Print het overzicht voor deze fold
    for naam, p in zip(features_in_fold, p_waarden_top10): # Loop door de gekozen features en hun p-waarden, print ze netjes uit
        status = "SIG" if p < 0.05 else "NIET SIG"
        print(f"Feature: {naam[:30]:<30} | p-waarde: {p:.5f} | {status}")
    print("-" * 30)

stabiele_features = set.intersection(*features_per_fold) # Bekijk welke features in alle 5 folds in de top 10 staan

print(f"Aantal features in alle 5 folds: {len(stabiele_features)}")
print(list(stabiele_features))


