import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC, SVC

from worcliver.load_data import load_data


# Use a fixed random state so results are reproducible across runs.
RANDOM_STATE = 42

# Use all available CPU cores for the cross-validation searches, useful with nested CV (a lot of hyperparameter combinations) to make faster
N_JOBS = -1

# Define the number of folds once so it is easy to reuse consistently.
N_OUTER_SPLITS = 5

# Define the number of inner folds used for hyperparameter tuning.
N_INNER_SPLITS = 3

# Define how many random hyperparameter combinations should be tested.
N_RANDOM_SEARCH_ITERATIONS = 20

# Store the ROC plot filename in one place to avoid hard-coded repetition.
ROC_OUTPUT_PATH = "roc_curve_nested_cv_svm.png"


class CorrUnivariateOptimizationSelector(BaseEstimator, TransformerMixin):
    # This custom transformer performs three feature-selection steps in sequence.
    def __init__(
        self,
        corr_threshold=0.9,
        k_univariate=20,
        n_features_to_select=5,
        rfe_estimator_c=1.0,
        rfe_step=1,
    ):
        # Save the maximum allowed absolute Spearman correlation between features.
        self.corr_threshold = corr_threshold
        # Save how many features should remain after univariate filtering.
        self.k_univariate = k_univariate
        # Save how many features RFE should keep at the end.
        self.n_features_to_select = n_features_to_select
        # Save the C value for the LinearSVC used inside RFE.
        self.rfe_estimator_c = rfe_estimator_c
        # Save how many features RFE removes at each step.
        self.rfe_step = rfe_step

    # Convert incoming data to a DataFrame so feature names are preserved.
    def _to_dataframe(self, X):
        # If the input is already a DataFrame, return a safe copy.
        if isinstance(X, pd.DataFrame):
            return X.copy()

        # If we already know the feature names, rebuild the DataFrame with them.
        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.feature_names_in_)

        # Otherwise, create a DataFrame with default integer column names.
        return pd.DataFrame(X)

    # Learn which features should be kept based on the training data only.
    def fit(self, X, y):
        # Ensure we work with a DataFrame so column-based feature tracking is easy.
        X_df = self._to_dataframe(X)

        # Store the original input feature names for later transforms.
        self.feature_names_in_ = X_df.columns.to_list()

        # Detect constant features because they contain no useful information.
        constant_mask = X_df.nunique(dropna=False) <= 1

        # Save the constant feature names so we can drop them later as well.
        self.constant_features_ = X_df.columns[constant_mask].to_list()

        # Remove constant features before any further selection step.
        X_non_constant = X_df.drop(columns=self.constant_features_, errors="ignore")

        # Compute the absolute Spearman correlation matrix.
        corr_matrix = X_non_constant.corr(method="spearman").abs().fillna(0)

        # Keep only the upper triangle so each feature pair is checked once.
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Mark features for removal when they correlate too strongly with another feature.
        self.correlation_features_ = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > self.corr_threshold)
        ]

        # Remove the highly correlated features.
        X_after_correlation = X_non_constant.drop(
            columns=self.correlation_features_,
            errors="ignore",
        )

        # Stop early if correlation filtering removed everything.
        if X_after_correlation.shape[1] == 0:
            raise ValueError("No features left after constant and correlation filtering.")

        # Limit k so SelectKBest never asks for more features than available.
        k_value = min(self.k_univariate, X_after_correlation.shape[1])

        # Create the ANOVA-based univariate feature selector.
        self.univariate_selector_ = SelectKBest(score_func=f_classif, k=k_value)

        # Fit the univariate selector on the current training fold.
        self.univariate_selector_.fit(X_after_correlation, y)

        # Save the feature names that survived the univariate step.
        self.univariate_features_ = X_after_correlation.columns[
            self.univariate_selector_.get_support()
        ].to_list()

        # Build the reduced dataset that will be used by RFE.
        X_after_univariate = X_after_correlation[self.univariate_features_]

        # Limit the final number of selected features to what is actually available.
        n_rfe_features = min(self.n_features_to_select, X_after_univariate.shape[1])

        # Stop early if there are no features left for the final optimization step.
        if n_rfe_features == 0:
            raise ValueError("No features left for optimization-based feature selection.")

        # Create the linear SVM that RFE uses to score feature importance.
        self.rfe_estimator_ = LinearSVC(
            C=self.rfe_estimator_c,
            dual=False,
            max_iter=10000,
            random_state=RANDOM_STATE,
        )

        # Create the RFE selector that recursively removes weak features.
        self.rfe_selector_ = RFE(
            estimator=self.rfe_estimator_,
            n_features_to_select=n_rfe_features,
            step=self.rfe_step,
        )

        # Fit RFE on the already reduced feature set.
        self.rfe_selector_.fit(X_after_univariate, y)

        # Save the final feature names selected by the full feature-selection pipeline.
        self.features_ = X_after_univariate.columns[
            self.rfe_selector_.get_support()
        ].to_list()

        # Return the fitted transformer to match scikit-learn conventions.
        return self

    # Apply the learned feature-selection steps to new data.
    def transform(self, X):
        # Convert the input to a DataFrame so the stored column names still work.
        X_df = self._to_dataframe(X)

        # Remove the constant features learned during fitting.
        X_non_constant = X_df.drop(columns=self.constant_features_, errors="ignore")

        # Remove the correlated features learned during fitting.
        X_after_correlation = X_non_constant.drop(
            columns=self.correlation_features_,
            errors="ignore",
        )

        # Keep only the features selected by the univariate step.
        X_after_univariate = X_after_correlation[self.univariate_features_]

        # Return only the final RFE-selected features.
        return X_after_univariate[self.features_]


# Build one reusable pipeline with feature selection, scaling, and SVM classification.
def build_pipeline():
    # Return a pipeline so every fold applies the same sequence of steps.
    return Pipeline(
        [
            (
                "feat_select",
                CorrUnivariateOptimizationSelector(
                    corr_threshold=0.9,
                    k_univariate=20,
                    n_features_to_select=5,
                    rfe_estimator_c=1.0,
                ),
            ),
            ("scaler", RobustScaler()),
            ("clf", SVC()),
        ]
    )


# Summarize the best hyperparameters by taking the most common value per setting.
def summarize_params(best_params_per_fold):
    # Create an empty dictionary that will hold the final summary.
    summary = {}

    # Loop over each hyperparameter key that appeared in the first fold.
    for key in best_params_per_fold[0]:
        # Collect that hyperparameter value from every outer fold.
        values = [params[key] for params in best_params_per_fold]

        # Keep the most frequently chosen value across folds.
        summary[key] = Counter(values).most_common(1)[0][0]

    # Return the summarized hyperparameter dictionary.
    return summary


# Summarize how stable the selected features are across outer folds.
def summarize_feature_stability(features_per_fold, n_folds):
    # Flatten all selected features into one long list.
    all_selected_features = [
        feature for fold_features in features_per_fold for feature in fold_features
    ]

    # Count how often each feature was selected.
    feature_counts = Counter(all_selected_features)

    # First prefer features that appeared in every outer fold.
    consensus_features = [
        feature for feature, count in feature_counts.items() if count == n_folds
    ]

    # If none appeared in all folds, keep features that appeared in almost all folds.
    if not consensus_features:
        consensus_features = [
            feature for feature, count in feature_counts.items() if count >= n_folds - 1
        ]

    # If stability is still low, fall back to the most frequently selected features.
    if not consensus_features:
        consensus_features = [
            feature
            for feature, _ in feature_counts.most_common(
                min(10, max(len(features_per_fold[0]), 1))
            )
        ]

    # Return both the raw counts and the simplified stable-feature summary.
    return feature_counts, consensus_features


# Save one ROC curve based on the outer-fold predictions from nested cross-validation.
def save_roc_curve(y_true, y_scores, roc_auc, output_path=ROC_OUTPUT_PATH):
    # Convert predictions into false-positive and true-positive rates.
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_scores)

    # Create the figure that will contain the ROC curve.
    plt.figure(figsize=(6, 6))

    # Draw the model ROC curve and show the resulting AUC in the legend.
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
        linewidth=2,
    )

    # Draw the diagonal reference line that represents random guessing.
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")

    # Label the x-axis.
    plt.xlabel("False Positive Rate")

    # Label the y-axis.
    plt.ylabel("True Positive Rate")

    # Add a clear title to explain what the figure represents.
    plt.title("Nested CV ROC Curve (SVM)")

    # Place the legend in the lower-right corner.
    plt.legend(loc="lower right")

    # Reduce clipping risk before saving the figure.
    plt.tight_layout()

    # Save the image to disk.
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # Close the figure so memory is released.
    plt.close()

    # Print the saved location for quick confirmation.
    print(f"Saved ROC curve to: {Path(output_path).resolve()}")


# Run nested cross-validation and return performance, selected features, and predictions.
def run_nested_cv(X, y):
    # Create the outer cross-validation loop used for unbiased model evaluation.
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Create the inner cross-validation loop used for hyperparameter tuning.
    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Define the search space for feature selection and SVM hyperparameters.
    param_dist = {
        "feat_select__corr_threshold": [0.8, 0.85, 0.9],
        "feat_select__k_univariate": [10, 15, 20, 30],
        "feat_select__n_features_to_select": [3, 5, 8, 10],
        "feat_select__rfe_estimator_c": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"],
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.1, 0.01, 0.001],
    }

    # Store the true labels from each outer validation fold.
    all_y_outer = []

    # Store the decision scores from each outer validation fold.
    all_scores_outer = []

    # Store the ROC-AUC of each outer fold separately.
    outer_fold_aucs = []

    # Store the selected features per fold so stability can be summarized later.
    features_per_fold = []

    # Store the best hyperparameters found in each fold.
    best_params_per_fold = []

    # Inform the user that the expensive nested CV process is starting.
    print("Start nested cross-validation...")

    # Loop over the outer folds one by one.
    for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        # Print which outer fold is currently being processed.
        print(f"\n--- Outer fold {fold_idx} ---")

        # Create the outer training subset for this fold.
        X_outer_train = X.iloc[outer_train_idx].copy()

        # Create the outer validation subset for this fold.
        X_outer_val = X.iloc[outer_val_idx].copy()

        # Create the training labels for this fold.
        y_outer_train = y.iloc[outer_train_idx]

        # Create the validation labels for this fold.
        y_outer_val = y.iloc[outer_val_idx]

        # Set up the randomized hyperparameter search inside the inner CV loop.
        search = RandomizedSearchCV(
            estimator=build_pipeline(),
            param_distributions=param_dist,
            n_iter=N_RANDOM_SEARCH_ITERATIONS,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
        )

        # Fit the full search process on the current outer training fold.
        search.fit(X_outer_train, y_outer_train)

        # Extract the best pipeline found inside the current outer fold.
        best_model = search.best_estimator_

        # Access the feature selector inside the best pipeline for reporting.
        selector = best_model.named_steps["feat_select"]

        # Save the best hyperparameters of this fold.
        best_params_per_fold.append(search.best_params_)

        # Save the selected final features of this fold.
        features_per_fold.append(selector.features_)

        # Generate decision scores for the unseen outer validation data.
        scores_outer = best_model.decision_function(X_outer_val)

        # Collect the true labels for the global nested ROC-AUC.
        all_y_outer.extend(y_outer_val)

        # Collect the validation scores for the global nested ROC-AUC.
        all_scores_outer.extend(scores_outer)

        # Print the best hyperparameters of the current fold.
        print(f"Best params: {search.best_params_}")

        # Print how many features remain after each selection step.
        print(
            "Features: "
            f"{X_outer_train.shape[1]} -> "
            f"{X_outer_train.shape[1] - len(selector.constant_features_)} (after constant) -> "
            f"{X_outer_train.shape[1] - len(selector.constant_features_) - len(selector.correlation_features_)} (after correlation) -> "
            f"{len(selector.univariate_features_)} (after univariate) -> "
            f"{len(selector.features_)} (after optimization)"
        )

        # Print the final selected features for this fold.
        print(f"Selected features: {selector.features_}")

        # Compute the ROC-AUC for this single outer fold.
        outer_fold_auc = roc_auc_score(y_outer_val, scores_outer)

        # Save the fold-specific ROC-AUC for later averaging.
        outer_fold_aucs.append(outer_fold_auc)

        # Print the fold ROC-AUC for transparency.
        print(f"Outer fold ROC-AUC: {outer_fold_auc:.3f}")

    # Compute the overall nested CV ROC-AUC using all outer-fold predictions together.
    nested_auc = roc_auc_score(all_y_outer, all_scores_outer)

    # Compute the mean ROC-AUC over the outer folds.
    outer_auc_mean = float(np.mean(outer_fold_aucs))

    # Compute the standard deviation of the outer-fold ROC-AUC values.
    outer_auc_std = float(np.std(outer_fold_aucs))

    # Summarize how often features were selected across folds.
    feature_counts, consensus_features = summarize_feature_stability(
        features_per_fold,
        outer_cv.get_n_splits(),
    )

    # Summarize the most common best hyperparameter values across folds.
    final_params = summarize_params(best_params_per_fold)

    # Return all important results from nested CV.
    return (
        nested_auc,
        outer_auc_mean,
        outer_auc_std,
        feature_counts,
        consensus_features,
        final_params,
        all_y_outer,
        all_scores_outer,
    )


# Run a regular 5-fold CV with one fixed pipeline configuration.
def run_regular_cv(X, y, final_params):
    # Build a new pipeline that will use the summarized best hyperparameters.
    final_pipeline = build_pipeline()

    # Apply the summarized best hyperparameters to the pipeline.
    final_pipeline.set_params(**final_params)

    # Create the regular 5-fold CV splitter.
    regular_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Evaluate the summarized pipeline with standard 5-fold cross-validation.
    regular_cv_scores = cross_val_score(
        final_pipeline,
        X,
        y,
        cv=regular_cv,
        scoring="roc_auc",
        n_jobs=N_JOBS,
    )

    # Return the mean and standard deviation of the regular 5-fold ROC-AUC.
    return float(regular_cv_scores.mean()), float(regular_cv_scores.std())


# Load the data, run the evaluations, print results, and save the ROC curve.
def main():
    # Load the dataset with the project-specific helper.
    data = load_data()

    # Keep only numeric columns because the model expects numeric input.
    X = data.select_dtypes(include=[np.number]).copy()

    # Convert the target labels to binary integers.
    y = data["label"].map({"benign": 0, "malignant": 1})

    # Run nested cross-validation on the full dataset.
    (
        nested_auc,
        nested_outer_mean,
        nested_outer_std,
        feature_counts,
        consensus_features,
        final_params,
        all_y_outer,
        all_scores_outer,
    ) = run_nested_cv(X, y)

    # Run a separate regular 5-fold CV using the summarized hyperparameters.
    regular_cv_mean, regular_cv_std = run_regular_cv(X, y, final_params)

    # Print a separator for the feature-stability summary.
    print("\n" + "=" * 40)

    # Explain that the next output is about feature stability.
    print("Feature stability")

    # Print how many unique features were selected across all outer folds.
    print(f"Unique selected features: {len(feature_counts)}")

    # Print the most stable features across folds.
    print(f"Consensus/stable features: {consensus_features}")

    # Print a separator for the performance metrics.
    print("\n" + "=" * 40)

    # Print the global nested CV ROC-AUC based on all outer-fold predictions.
    print(f"Nested CV ROC-AUC: {nested_auc:.3f}")

    # Print the average and spread of the outer-fold ROC-AUC values.
    print(
        f"Nested 5-fold ROC-AUC per outer fold: "
        f"{nested_outer_mean:.3f} +/- {nested_outer_std:.3f}"
    )

    # Print the summarized best hyperparameters.
    print(f"Most common best params over outer folds: {final_params}")

    # Print the separate regular 5-fold CV result.
    print(f"5-fold CV ROC-AUC: {regular_cv_mean:.3f} +/- {regular_cv_std:.3f}")

    # Save the ROC curve based on the nested CV outer-fold predictions.
    save_roc_curve(all_y_outer, all_scores_outer, nested_auc)


# Run the script only when this file is executed directly.
if __name__ == "__main__":
    # Start the full evaluation workflow.
    main()
