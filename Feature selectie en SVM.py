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
        top_indices = sorted_sig_indices[:20]
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
    print("\n" + "="*40)
    print("RESULTATEN MET SVM & STABIELE FEATURES")
    print("="*40)
    print(classification_report(all_y_test, all_y_pred))

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_y_test, all_y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title(f'SVM Confusion Matrix ({len(stabiele_features)} Features)')
    plt.ylabel('Werkelijke Diagnose')
    plt.xlabel('Voorspelling')
    plt.show()