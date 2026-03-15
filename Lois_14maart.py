#eerste probeersel
#%% Data loading functions. Uncomment the one you want to use
import pandas as pd
import numpy as np
from worcliver.load_data import load_data
import matplotlib.pyplot as plt

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

#Je wilt even kijken of het aantal malignant en benign tumoren een beetje vergelijkbaar is
#als je bijv 80% van een label hebt kun je met een model wat altijd dat zegt al een hoge accuracy bereiken
#print(data['label'].value_counts())
#print(data['label'].value_counts(normalize=True))

#check the missing values
#print(data.isnull().sum())
#no missing values!

from sklearn.model_selection import train_test_split

# Separate features (X) and labels (y)
X = data.drop(columns=['label'])
y = data['label'].map({'benign':0, 'malignant':1})

# Split the data into 80% training data and 20% test data
# Stratification keeps the class balance similar in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42 #this because you want the experiment to be reproducible, everytime you run it give the exact same results
)

# Print the shapes to verify the split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train distribution:")
print(y_train.value_counts(normalize=True))
print("y_test distribution:")
print(y_test.value_counts(normalize=True))

#going further with the X_train and y_train, split dataset in 5 folds (5-fold cross validation)
#per fold you want a sub-train part and a validation part (which is your test part within the cross-validation)
from sklearn.model_selection import StratifiedKFold

# Create a 5-fold cross-validation object
# StratifiedKFold keeps the class ratio similar in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#%% 1) FEATURE SELECTION

from scipy.stats import ttest_ind
# Create a list to store the selected features of each fold
selected_features_per_fold = []
# Lists to store curves of all folds
train_accuracy_curves_per_fold = []
val_accuracy_curves_per_fold = []
loss_curves_per_fold = []

# Number of epochs
n_epochs = 80

# Loop over all cross-validation folds
for fold_number, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):

    print(f"\nFold {fold_number}")

    # Create the sub-train and validation sets for this fold
    X_train_fold = X_train.iloc[train_idx] #take the rows of the particular fold out of the trainingset
    y_train_fold = y_train.iloc[train_idx]

    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Because p > N, we have a risk of overfitting.
    # Therefore, we reduce the number of features using univariate feature selection.
    # We perform a t-test for each feature on the training fold only.

    # Split the training fold into the two classes
    X_train_benign = X_train_fold[y_train_fold == 0]
    X_train_malignant = X_train_fold[y_train_fold == 1]

    # Create empty lists to store the t-statistics and p-values
    t_values = []
    p_values = []

    # Loop over all feature columns in the training fold
    for col in X_train_fold.columns:

        # Perform an independent t-test for the current feature
        # This compares the mean feature value between the benign and malignant groups
        t_stat, p_val = ttest_ind(
            X_train_benign[col].values,
            X_train_malignant[col].values,
            equal_var=False,   # Use Welch's t-test because equal variance cannot be assumed
            nan_policy='omit'  # Ignore missing values if they are present
        )

        # Store the t-statistic and p-value
        t_values.append(t_stat)
        p_values.append(p_val)

    # Create a DataFrame with the feature selection results for this fold
    feature_results = pd.DataFrame({
        'feature': X_train_fold.columns,   # Name of the feature
        't_statistic': t_values,           # T-test statistic
        'p_value': p_values,               # P-value of the t-test
        'abs_t': np.abs(t_values)          # Absolute t-statistic, useful for ranking
    })

    # Sort features by smallest p-value first
    # If p-values are similar, sort by largest absolute t-statistic
    feature_results = feature_results.sort_values(
        by=['p_value', 'abs_t'],
        ascending=[True, False]
    )
    # Select the top 14 best features for this fold
    top14 = feature_results.head(14)
    top14_features = top14['feature'].tolist()

    # Store the selected features of this fold
    selected_features_per_fold.append(top14_features)

    # Print the selected features and their statistics for this fold
    print("Selected feature:")
    print(top14[['feature', 't_statistic', 'p_value']])
    print("-" * 50)

    # Reduce datasets to selected features
    X_train_fold_selected = X_train_fold[top14_features]
    X_val_fold_selected = X_val_fold[top14_features]

    #%% 2) SCALING
    from sklearn.preprocessing import StandardScaler
    # Create a scaler object
    scaler = StandardScaler()

    # Fit the scaler only on the training fold
    # This calculates the mean and standard deviation of each feature
    X_train_fold_scaled = scaler.fit_transform(X_train_fold_selected)

    # Apply the same scaling to the validation fold
    X_val_fold_scaled = scaler.transform(X_val_fold_selected)

    #%% 3) CLASSIFICATION NN
    from sklearn.neural_network import MLPClassifier

    # Create the neural network classifier
    model = MLPClassifier(
        hidden_layer_sizes=(10,),  # One hidden layer with 10 neurons, these neurons make combinations of the 14 input features
        solver= "sgd",             # use stochastic gradient descent
        learning_rate_init=0.01,        #learning rate
        random_state=42            # Ensure reproducibility
    )

    train_scores = []
    val_scores = []
    loss_scores =[]

    # Train step-by-step using partial_fit so we can track performance
    for epoch in range(n_epochs):

        # Update the model weights
        model.partial_fit(X_train_fold_scaled, y_train_fold, classes=np.array([0,1])) #because you want to have control of each epoch, partial_fit, so you can measure accuracy of each epoch

        # Calculate accuracy on training and validation data
        train_scores.append(model.score(X_train_fold_scaled, y_train_fold))
        val_scores.append(model.score(X_val_fold_scaled, y_val_fold))
        loss_scores.append(model.loss_)

    # Store curves for this fold
    train_accuracy_curves_per_fold.append(train_scores)
    val_accuracy_curves_per_fold.append(val_scores)
    loss_curves_per_fold.append(loss_scores)

# Calculate mean curves across folds
mean_train_curve = np.mean(train_accuracy_curves_per_fold, axis=0)
mean_val_curve = np.mean(val_accuracy_curves_per_fold, axis=0)
mean_loss_curve = np.mean(loss_curves_per_fold, axis=0)

# Plot mean curves
plt.figure(figsize=(8, 5))
plt.plot(mean_train_curve, label="Mean training accuracy")
plt.plot(mean_val_curve, label="Mean validation accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Mean training vs validation accuracy across folds")

plt.legend()
plt.show()

# Plot mean loss across folds
plt.figure(figsize=(8, 5))
plt.plot(mean_loss_curve, label="Mean training loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Mean training loss across folds")
plt.legend()
plt.show()