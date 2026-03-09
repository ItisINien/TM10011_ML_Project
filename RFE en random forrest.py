import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from worcliver.load_data import load_data
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE # Nieuwe import
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- STAP 1: Dataset laden ---
data = load_data()
X = data.select_dtypes(include=[np.number])
y = data['label']

# --- STAP 2: K-Fold voorbereiden ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_y_test = []
all_y_pred = []

print("Starten met Random Forest + RFE (Recursive Feature Elimination)...")

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # --- STAP 3 & 4: Schalen ---
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- STAP 5: FEATURE SELECTIE (RFE) ---
    # We gebruiken een Random Forest om de features te selecteren
    # step=5 betekent dat hij er per ronde 5 weggooit (gaat sneller)
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    selector = RFE(estimator, n_features_to_select=20, step=5)
    
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # --- STAP 6: MODEL TRAINEN ---
    # Nu trainen we het definitieve model op de 20 overgebleven features
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_selected, y_train)
    
    # --- STAP 7: EVALUEREN ---
    y_pred = clf.predict(X_test_selected)
    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)
    
    print(f"Fold {i+1} afgerond.")

# --- FINALE RESULTATEN ---
print("\n" + "="*40)
print("RESULTATEN: RANDOM FOREST + RFE")
print("="*40)
print(classification_report(all_y_test, all_y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_y_test, all_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('Confusion Matrix: RF + RFE Selectie')
plt.ylabel('Werkelijkheid')
plt.xlabel('Voorspelling')
plt.show()