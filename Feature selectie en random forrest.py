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

# --- STAP 1: Dataset laden ---
data = load_data()
X = data.select_dtypes(include=[np.number])
y = data['label']

# --- STAP 2: K-Fold voorbereiden ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# We maken een lijst om de top-features per fold in op te slaan
features_per_fold = []

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
        top_indices = sorted_sig_indices[:20]
        features_per_fold.append(set(X.columns[top_indices]))

# Vind de doorsnede: welke features staan in ELKE fold in de top 20?
stabiele_features = list(set.intersection(*features_per_fold))

print(f"\nAantal stabiele features gevonden (in alle 5 folds): {len(stabiele_features)}")
print(f"Features: {stabiele_features}\n")

if len(stabiele_features) == 0:
    print("FOUT: Geen enkele feature komt in alle folds voor. Probeer de top 50?")
    # Optioneel: stop hier of verruim de selectie
else:
    # --- TWEEDE LOOP: TRAINING MET ALLEEN STABIELE FEATURES ---
    all_y_test = []
    all_y_pred = []

    print("Stap 2: Model trainen op alleen stabiele features...")
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Gebruik alleen de stabiele kolommen
        X_train_stabiel = X_train[stabiele_features]
        X_test_stabiel = X_test[stabiele_features]
        
        # Opnieuw schalen (binnen de fold)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_stabiel)
        X_test_scaled = scaler.transform(X_test_stabiel)
        
        # Model trainen
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        # Voorspellen
        y_pred = clf.predict(X_test_scaled)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    # --- FINALE RESULTATEN ---
    print("\n" + "="*40)
    print("RESULTATEN MET STABIELE FEATURES")
    print("="*40)
    print(classification_report(all_y_test, all_y_pred))

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_y_test, all_y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title(f'Confusion Matrix ({len(stabiele_features)} Stabiele Features)')
    plt.ylabel('Werkelijke Diagnose')
    plt.xlabel('Voorspelling')
    plt.show()