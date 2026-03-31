# %%
# Dit importeert standaardhulpmiddelen voor paden, tellen en omgeving.
import os
from collections import Counter
from pathlib import Path

# Dit importeert de basisbibliotheken voor numerieke data en tabellen.
import numpy as np
import pandas as pd

# Dit zet tijdelijke cachemappen zodat matplotlib ook in een beperkte omgeving werkt.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

# Dit gebruiken we om ROC-curves op te slaan.
import matplotlib.pyplot as plt

# Dit probeert SHAP te laden; als dat niet lukt slaan we SHAP rustig over.
try:
    import shap
except ImportError:
    shap = None

# Dit importeert alle sklearn-onderdelen voor selectie, modelleren en evaluatie.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer, roc_auc_score, roc_curve
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from worcliver.load_data import load_data


# Dit zijn vaste instellingen zodat alle runs reproduceerbaar en consistent blijven.
RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
N_RANDOM_SEARCH_ITERATIONS = 20
N_JOBS = 1
F_BETA = 2
ROC_OUTPUT_PATH = "roc_curve_nested_cv_lg.png"
MIN_FEATURE_FOLD_COUNT = 3
SHAP_OUTPUT_DIR = "shap_outputs_nested_cv_lg"


class CorrAndSelect(BaseEstimator, TransformerMixin):
    # Dit combineert het verwijderen van constante/gecorreleerde features met univariate selectie.
    def __init__(self, k=200, corr_threshold=0.9):
        self.k = k
        self.corr_threshold = corr_threshold

    def _to_dataframe(self, X):
        # Dit zorgt dat alle vervolgstappen met DataFrames en feature-namen kunnen werken.
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.feature_names_in_)
        return pd.DataFrame(X)

    def fit(self, X, y):
        # Dit leert welke features we willen behouden op basis van de trainingsdata.
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = X_df.columns.to_list()

        # Dit verwijdert features zonder variatie, omdat die niets bijdragen aan het model.
        constant_mask = X_df.nunique(dropna=False) <= 1
        self.constant_features_ = X_df.columns[constant_mask].to_list()
        self.near_constant_features_ = []
        self.removed_low_variation_features_ = self.constant_features_
        X_non_constant = X_df.drop(
            columns=self.removed_low_variation_features_,
            errors="ignore",
        )

        # Dit zoekt sterk gecorreleerde features zodat we redundante informatie kunnen schrappen.
        corr_matrix = X_non_constant.corr(method="spearman").abs().fillna(0)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [
            col for col in upper.columns if any(upper[col] > self.corr_threshold)
        ]

        X_filtered = X_non_constant.drop(columns=self.to_drop_, errors="ignore")

        if X_filtered.shape[1] == 0:
            raise ValueError("No features left after constant/correlation filtering.")

        # Dit houdt de best scorende features over via univariate ANOVA-selectie.
        k_value = min(self.k, X_filtered.shape[1])
        self.selector_ = SelectKBest(score_func=f_classif, k=k_value)
        self.selector_.fit(X_filtered, y)

        # Dit bewaart zowel de gekozen features als tussentijdse aantallen voor rapportage.
        self.features_ = X_filtered.columns[self.selector_.get_support()].to_list()
        self.n_input_features_ = X_df.shape[1]
        self.n_after_constant_ = X_non_constant.shape[1]
        self.n_after_correlation_ = X_filtered.shape[1]
        self.n_after_univariate_ = len(self.features_)
        return self

    def transform(self, X):
        # Dit past precies dezelfde filtering en selectie toe op nieuwe data.
        X_df = self._to_dataframe(X)
        X_non_constant = X_df.drop(
            columns=self.removed_low_variation_features_,
            errors="ignore",
        )
        X_filtered = X_non_constant.drop(columns=self.to_drop_, errors="ignore")
        transformed = self.selector_.transform(X_filtered)
        return pd.DataFrame(transformed, columns=self.features_, index=X_df.index)


class DataFrameRobustScaler(BaseEstimator, TransformerMixin):
    # Dit is een wrapper zodat schalen DataFrame-structuur en feature-namen behoudt.
    def __init__(self):
        self.scaler_ = RobustScaler()

    def _to_dataframe(self, X):
        # Dit zorgt dat de scaler altijd met een DataFrame werkt.
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.feature_names_in_)
        return pd.DataFrame(X)

    def fit(self, X, y=None):
        # Dit leert de schaalparameters op de trainingsdata.
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = X_df.columns.to_list()
        self.scaler_.fit(X_df, y)
        return self

    def transform(self, X):
        # Dit schaalt nieuwe data en geeft die weer terug als DataFrame.
        X_df = self._to_dataframe(X)
        transformed = self.scaler_.transform(X_df)
        return pd.DataFrame(transformed, columns=self.feature_names_in_, index=X_df.index)


def build_pipeline():
    # Dit model gebruikt RFE alleen als selectiestap binnen de pipeline.
    lr_for_rfe = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    # Dit is de uiteindelijke logistic regression die de voorspelling maakt.
    final_lr = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )

    # Dit bouwt de volledige pipeline: filteren, schalen, optimalisatie-gebaseerde selectie en classificatie.
    return Pipeline(
        [
            ("feat_select", CorrAndSelect(k=200, corr_threshold=0.9)),
            ("scaler", DataFrameRobustScaler()),
            (
                "rfe",
                RFE(
                    estimator=lr_for_rfe,
                    n_features_to_select=10,
                    step=0.1,
                ),
            ),
            ("clf", final_lr),
        ]
    )


def compute_fbeta_from_proba(y_true, y_proba, beta=F_BETA, threshold=0.5):
    # Dit zet kansen om naar klasses en berekent daarna de F-beta score.
    y_pred = (np.array(y_proba) >= threshold).astype(int)
    return float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))


def summarize_params(best_params_per_fold):
    # Dit kiest per hyperparameter de meest voorkomende beste waarde over de outer folds.
    summary = {}
    for key in best_params_per_fold[0]:
        values = [params[key] for params in best_params_per_fold]
        summary[key] = Counter(values).most_common(1)[0][0]
    return summary


def summarize_feature_stability(features_per_fold):
    # Dit telt hoe vaak elke feature geselecteerd werd over alle outer folds.
    all_features = [feature for fold_features in features_per_fold for feature in fold_features]
    return Counter(all_features)


def save_roc_curve(y_true, y_scores, roc_auc, output_path=ROC_OUTPUT_PATH):
    # Dit berekent de ROC-curve en slaat die op als png-bestand.
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
        linewidth=2,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Nested CV ROC Curve (Logistic Regression)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve to: {Path(output_path).resolve()}")


def run_nested_shap_summary(nested_shap_payload, output_dir=SHAP_OUTPUT_DIR):
    # Sla SHAP over als de dependency ontbreekt.
    if shap is None:
        print("\nNested SHAP summary skipped: package 'shap' is not installed.")
        return

    # Zonder fold-resultaten kunnen we geen nested SHAP-samenvatting maken.
    if not nested_shap_payload:
        print("\nNested SHAP summary skipped: no SHAP payload available.")
        return

    # Neem alle features die in minstens een outer fold voorkomen.
    all_features = sorted(
        {
            feature
            for fold_payload in nested_shap_payload
            for feature in fold_payload["feature_names"]
        }
    )
    if not all_features:
        print("\nNested SHAP summary skipped: no features available.")
        return

    # Maak de outputmap aan voor de csv-samenvatting.
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    combined_shap_frames = []
    for fold_payload in nested_shap_payload:
        # Lijn alle folds uit op dezelfde featurekolommen; ontbrekende features krijgen 0.
        shap_frame = pd.DataFrame(
            fold_payload["shap_values"],
            columns=fold_payload["feature_names"],
            index=fold_payload["index"],
        ).reindex(columns=all_features, fill_value=0.0)
        combined_shap_frames.append(shap_frame)

    # Voeg SHAP-resultaten uit alle outer folds samen voor een nested CV-overzicht.
    combined_shap = pd.concat(combined_shap_frames, axis=0).sort_index()

    # Gebruik de gemiddelde absolute SHAP-waarde als globale feature importance.
    importance_df = (
        pd.DataFrame(
            {
                "feature": all_features,
                "mean_abs_shap": np.abs(combined_shap.to_numpy()).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Bewaar alleen de compacte samenvatting als csv.
    importance_df.to_csv(output_path / "nested_shap_importance.csv", index=False)

    print("\n" + "=" * 40)
    print("Nested CV SHAP mean absolute importance")
    print(importance_df.to_string(index=False))
    print(f"Saved nested SHAP summary to: {output_path.resolve()}")


def run_nested_cv(X_trainval, y_trainval):
    # Dit maakt een outer en inner cross-validation voor eerlijke nested modelselectie.
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Dit definieert welke hyperparameters we binnen de nested search gaan uitproberen.
    param_dist = {
        "feat_select__k": [10, 15, 20, 30],
        "feat_select__corr_threshold": [0.8, 0.85, 0.9],
        "rfe__n_features_to_select": [6, 8, 10, 12],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "rfe__estimator__C": [0.01, 0.1, 1, 10],
    }

    # Dit verzamelt alle fold-uitkomsten zodat we achteraf nested prestaties kunnen samenvatten.
    all_y_outer = []
    all_y_outer_proba = []
    outer_fold_aucs = []
    outer_fold_f2s = []
    features_per_fold = []
    best_params_per_fold = []
    # Hier bewaren we per outer fold de SHAP-resultaten van de trainingsdata.
    nested_shap_payload = []

    print("Start nested cross-validation on the 80% train/validation set...")

    for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(
        outer_cv.split(X_trainval, y_trainval),
        start=1,
    ):
        print(f"\n--- Outer fold {fold_idx} ---")

        # Dit splitst per outer fold in een trainings- en validatiedeel.
        X_outer_train = X_trainval.iloc[outer_train_idx].copy()
        X_outer_val = X_trainval.iloc[outer_val_idx].copy()
        y_outer_train = y_trainval.iloc[outer_train_idx]
        y_outer_val = y_trainval.iloc[outer_val_idx]

        # Dit zoekt binnen de inner CV naar de beste hyperparameters op basis van F2.
        search = RandomizedSearchCV(
            estimator=build_pipeline(),
            param_distributions=param_dist,
            n_iter=N_RANDOM_SEARCH_ITERATIONS,
            cv=inner_cv,
            # Stem de hyperparameters af op F2 zodat recall zwaarder meetelt.
            scoring=make_scorer(fbeta_score, beta=F_BETA, zero_division=0),
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
        )
        search.fit(X_outer_train, y_outer_train)

        # Dit pakt het beste model uit deze outer fold en leest de geselecteerde features uit.
        best_model = search.best_estimator_
        selector = best_model.named_steps["feat_select"]
        rfe = best_model.named_steps["rfe"]

        # Dit gebruikt kansen voor klasse 1 als continue score voor AUC en classificatie-evaluatie.
        y_outer_proba = best_model.predict_proba(X_outer_val)[:, 1]
        fold_auc = roc_auc_score(y_outer_val, y_outer_proba)
        fold_f2 = compute_fbeta_from_proba(y_outer_val, y_outer_proba)

        # Dit bepaalt welke features na univariate selectie en RFE echt overblijven.
        selected_features = np.array(selector.features_)[rfe.support_].tolist()

        # Dit bewaart alle foldresultaten zodat we die later samen kunnen vatten.
        all_y_outer.extend(y_outer_val)
        all_y_outer_proba.extend(y_outer_proba)
        outer_fold_aucs.append(fold_auc)
        outer_fold_f2s.append(fold_f2)
        features_per_fold.append(selected_features)
        best_params_per_fold.append(search.best_params_)

        # Bereken SHAP op de trainingsdata van de outer fold om het model binnen
        # nested CV te interpreteren zonder de testset te gebruiken.
        if shap is not None:
            scaled_train = pd.DataFrame(
                best_model.named_steps["scaler"].transform(
                    selector.transform(X_outer_train)
                ),
                columns=selector.features_,
                index=X_outer_train.index,
            )
            final_features = np.array(selector.features_)[rfe.support_].tolist()
            scaled_train_final = pd.DataFrame(
                rfe.transform(scaled_train),
                columns=final_features,
                index=X_outer_train.index,
            )

            # Neem een kleine achtergrondset voor een stabiele en snellere SHAP-berekening.
            background_size = min(100, len(scaled_train_final))
            background = scaled_train_final.sample(
                background_size, random_state=RANDOM_STATE
            )
            explainer = shap.LinearExplainer(best_model.named_steps["clf"], background)
            shap_values = explainer(scaled_train_final)

            # Bewaar per fold alleen wat nodig is voor de uiteindelijke nested samenvatting.
            nested_shap_payload.append(
                {
                    "feature_names": final_features,
                    "shap_values": shap_values.values,
                    "index": scaled_train_final.index,
                }
            )

        print(f"Best params: {search.best_params_}")
        print(
            "Features per step: "
            f"{selector.n_input_features_} -> "
            f"{selector.n_after_constant_} (after constant) -> "
            f"{selector.n_after_correlation_} (after correlation) -> "
            f"{selector.n_after_univariate_} (after univariate) -> "
            f"{len(selected_features)} (after RFE)"
        )
        print(f"Selected features: {selected_features}")
        print(f"Outer fold ROC-AUC: {fold_auc:.3f}")
        print(f"Outer fold F{F_BETA}-score: {fold_f2:.3f}")

    # Bereken de totale nested prestatie over alle outer-fold voorspellingen samen.
    nested_auc = roc_auc_score(all_y_outer, all_y_outer_proba)
    nested_f2 = compute_fbeta_from_proba(all_y_outer, all_y_outer_proba)
    feature_counts = summarize_feature_stability(features_per_fold)
    common_params = summarize_params(best_params_per_fold)

    # Dit geeft alle nested resultaten terug in een dictionary voor verdere analyse en vergelijking.
    return {
        "nested_auc": float(nested_auc),
        "nested_f2": float(nested_f2),
        "outer_fold_aucs": outer_fold_aucs,
        "outer_fold_f2s": outer_fold_f2s,
        "feature_counts": feature_counts,
        "common_params": common_params,
        "best_params_per_fold": best_params_per_fold,
        "features_per_fold": features_per_fold,
        "all_y_outer": np.array(all_y_outer),
        "all_y_outer_proba": np.array(all_y_outer_proba),
        "nested_shap_payload": nested_shap_payload,
    }


def run_regular_5fold_cv(X_trainval, y_trainval, common_params):
    # Dit voert nog een gewone 5-fold CV uit met de meest gekozen hyperparameters.
    pipeline = build_pipeline()
    pipeline.set_params(**common_params)

    regular_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Dit meet hoe stabiel de uiteindelijke pipeline scoort op ROC-AUC en F2.
    roc_auc_scores = cross_val_score(
        pipeline,
        X_trainval,
        y_trainval,
        cv=regular_cv,
        scoring="roc_auc",
        n_jobs=N_JOBS,
    )
    f2_scores = cross_val_score(
        pipeline,
        X_trainval,
        y_trainval,
        cv=regular_cv,
        scoring=make_scorer(fbeta_score, beta=F_BETA, zero_division=0),
        n_jobs=N_JOBS,
    )

    # Dit geeft de gemiddelde score en spreiding terug voor rapportage.
    return {
        "roc_auc_mean": float(np.mean(roc_auc_scores)),
        "roc_auc_std": float(np.std(roc_auc_scores)),
        "f2_mean": float(np.mean(f2_scores)),
        "f2_std": float(np.std(f2_scores)),
    }


def fit_final_model(X_trainval, y_trainval, common_params):
    # Dit traint het uiteindelijke model op alle train/validation-data samen.
    pipeline = build_pipeline()
    pipeline.set_params(**common_params)
    pipeline.fit(X_trainval, y_trainval)
    return pipeline


def main():
    # Dit laadt de data en houdt alleen numerieke features plus het label over.
    data = load_data()
    X = data.select_dtypes(include=[np.number]).copy()
    y = data["label"].map({"benign": 0, "malignant": 1})

    # Dit maakt alvast een vaste train/test-split, waarbij nested CV alleen op trainval draait.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Dit voert nested CV uit om de prestaties en hyperparameters eerlijk te schatten.
    nested_results = run_nested_cv(X_trainval, y_trainval)
    # Dit voert nog een gewone 5-fold CV uit met de gekozen hyperparameters.
    regular_cv_results = run_regular_5fold_cv(
        X_trainval,
        y_trainval,
        nested_results["common_params"],
    )

    print("\n" + "=" * 40)
    print("Summary over outer folds")
    print(f"Nested CV ROC-AUC: {nested_results['nested_auc']:.3f}")
    print(f"Nested CV F{F_BETA}-score: {nested_results['nested_f2']:.3f}")
    print(
        f"5-fold ROC-AUC: {regular_cv_results['roc_auc_mean']:.3f} +/- "
        f"{regular_cv_results['roc_auc_std']:.3f}"
    )
    print(
        f"5-fold F{F_BETA}-score: {regular_cv_results['f2_mean']:.3f} +/- "
        f"{regular_cv_results['f2_std']:.3f}"
    )
    print(f"Most common hyperparameters: {nested_results['common_params']}")

    hyperparameter_counts = {
        key: Counter(params[key] for params in nested_results["best_params_per_fold"])
        for key in nested_results["best_params_per_fold"][0]
    }
    print("Hyperparameter frequencies:")
    for key, counts in hyperparameter_counts.items():
        print(f"{key}: {dict(counts)}")

    print("Most common features:")
    stable_features = [
        (feature, count)
        for feature, count in nested_results["feature_counts"].most_common()
        if count >= MIN_FEATURE_FOLD_COUNT
    ]
    if stable_features:
        for feature, count in stable_features:
            print(f"{feature}: selected in {count}/{N_OUTER_SPLITS} folds")
    else:
        print(f"No features were selected in at least {MIN_FEATURE_FOLD_COUNT}/{N_OUTER_SPLITS} folds.")

    # Dit traint nog eenmaal het finale model op alle train/validation-data.
    final_pipeline = fit_final_model(
        X_trainval,
        y_trainval,
        nested_results["common_params"],
    )
    # Maak een compacte SHAP-samenvatting op basis van de trainingsdata uit de outer folds.
    run_nested_shap_summary(nested_results["nested_shap_payload"])
    # Bewaar de nested ROC-curve op basis van alle out-of-fold kansen samen.
    save_roc_curve(
        nested_results["all_y_outer"],
        nested_results["all_y_outer_proba"],
        nested_results["nested_auc"],
    )


if __name__ == "__main__":
    # Dit zorgt dat main() alleen draait als je dit bestand direct uitvoert.
    main()

# %%
def Logistic_Uni_opt(test=False):
    # Dit is de functie die Final.py gebruikt om LG-resultaten op te halen.
    data = load_data()
    X = data.select_dtypes(include=[np.number]).copy()
    y = data["label"].map({"benign": 0, "malignant": 1})

    # Dit maakt dezelfde vaste train/test-split als in main().
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Dit voert nested CV uit op alleen het train/validation-deel.
    nested_results = run_nested_cv(X_trainval, y_trainval)

    # Dit traint daarna het finale model met de meest gekozen hyperparameters.
    final_pipeline = fit_final_model(
        X_trainval,
        y_trainval,
        nested_results["common_params"],
    )

    # 5. Maak bij elke run de nested SHAP-samenvatting en ROC-curve aan.
    run_nested_shap_summary(nested_results["nested_shap_payload"])
    save_roc_curve(
        nested_results["all_y_outer"],
        nested_results["all_y_outer_proba"],
        nested_results["nested_auc"],
    )

    # Dit maakt de kernresultaten klaar die Final.py gebruikt voor modelvergelijking.
    results = {
        "nested_auc_lg": nested_results["nested_auc"],
        "nested_f2_lg": nested_results["nested_f2"],
        "fold_aucs_lg": nested_results["outer_fold_aucs"],
        "outer_fold_f2s_lg": nested_results["outer_fold_f2s"],
        "common_params_lg": nested_results["common_params"],
        "features_per_fold_lg": nested_results["features_per_fold"],
    }
    # Gebruik de klasse-1 kans als score voor ROC-AUC, F2 en eventuele ROC-plot in Final.py.
    test_scores = final_pipeline.predict_proba(X_test)[:, 1]

    if test:
        # Evalueer pas hier op de echte testset nadat het model is gekozen.
        test_auc = roc_auc_score(y_test, test_scores)
        test_pred = (test_scores > 0.5).astype(int)
        test_f2 = fbeta_score(y_test, test_pred, beta=2)

        # Geef ook de ruwe testscores en labels terug zodat Final.py zelf de ROC-curve kan maken.
        results["test_auc"] = float(test_auc)
        results["test_f2"] = float(test_f2)
        results["test_scores"] = test_scores
        results["y_test"] = y_test.to_numpy()

    return results
