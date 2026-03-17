# tweede probeersel

#%% Data loading functions. Uncomment the one you want to use
import pandas as pd
import numpy as np
from worcliver.load_data import load_data
import matplotlib.pyplot as plt

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

# You can inspect whether the classes are balanced
# If one class dominates, a model could get high accuracy by always predicting that class
#print(data['label'].value_counts())
#print(data['label'].value_counts(normalize=True))

# Check missing values
#print(data.isnull().sum())
# No missing values

from sklearn.model_selection import train_test_split

# Separate features (X) and labels (y)
X = data.drop(columns=['label'])
y = data['label'].map({'benign': 0, 'malignant': 1})

# Split the data into 80% training data and 20% test data
# Stratification keeps the class balance similar in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42  # This makes the split reproducible
)

# Print the shapes to verify the split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train distribution:")
print(y_train.value_counts(normalize=True))
print("y_test distribution:")
print(y_test.value_counts(normalize=True))

# Going further with X_train and y_train, split the training set into 5 folds
# Each fold has a sub-train part and a validation part
from sklearn.model_selection import StratifiedKFold

# Create a 5-fold cross-validation object
# StratifiedKFold keeps the class ratio similar in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#%% FEATURE SELECTION - Optimization-based feature selection with RFECV + Random Forest

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

# Scale the training data
# Scaling is not strictly required for Random Forest,
# but we do it for consistency in the pipeline
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

# Scale the test data using the scaler fitted on the training data only
# This prevents information from the test set leaking into training
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Define the model that will be used inside RFECV
# Random Forest can estimate feature importance
rf_model = RandomForestClassifier(
    n_estimators=200,   # Number of trees in the forest
    random_state=42,
    n_jobs=-1
)

# Create the RFECV object
# RFECV recursively removes the least important features
# and uses cross-validation to find the optimal feature subset
rfecv = RFECV(
    estimator=rf_model,
    step=1,                     # Remove one feature at a time
    cv=cv,                      # Use the 5-fold cross-validation defined above
    scoring='roc_auc',          # Evaluate performance using ROC-AUC
    min_features_to_select=1,
    n_jobs=-1
)

# Fit RFECV on the training data only
# This selects the best subset of features without touching the test set
rfecv.fit(X_train_scaled, y_train)

# Get the names of the selected features
selected_features_rfecv = X_train.columns[rfecv.support_].tolist()

# Keep only the selected features
X_train_selected = X_train_scaled[selected_features_rfecv]
X_test_selected = X_test_scaled[selected_features_rfecv]

# Print the optimization results
print("Optimal number of features:", rfecv.n_features_)
print("Selected features:")
for feature in selected_features_rfecv:
    print("-", feature)

# Plot the mean cross-validation score for each number of selected features
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
    rfecv.cv_results_["mean_test_score"],
    marker='o'
)
plt.xlabel("Number of selected features")
plt.ylabel("Mean cross-validation ROC-AUC")
plt.title("RFECV with Random Forest")
plt.grid(True)
plt.show()

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train classifier on selected features
rf_model.fit(X_train_selected, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test_selected)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Test accuracy:", accuracy)

# Confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Detailed metrics
print("Classification report:")
print(classification_report(y_test, y_pred))