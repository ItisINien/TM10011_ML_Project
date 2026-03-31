# %%
# Dit importeert de basisbibliotheken voor numerieke data, tabellen en bestandslocaties.
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# Dit gebruiken we om ROC- en eventueel SHAP-plots op te slaan.
import matplotlib.pyplot as plt

# Dit importeert sklearn-onderdelen voor featureselectie, evaluatie en modelleren.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

# Dit laadt de projectdata via de bestaande helper uit worcliver.
from worcliver.load_data import load_data

# Dit probeert SHAP te laden; als dat niet lukt slaan we SHAP-berekeningen over.
try:
    import shap
except ImportError:
    shap = None


# Dit zorgt dat alle splits en random keuzes reproduceerbaar blijven.
RANDOM_STATE = 42


class CorrUnivariateSelector(BaseEstimator, TransformerMixin):
    # Dit combineert constante-filtering, correlatiefiltering en univariate selectie in één stap.
    def __init__(
        self,
        corr_threshold=0.9,
        k_univariate=20,
        consensus_n_splits=5,
        consensus_min_fraction=0.6,
    ):
        self.corr_threshold = corr_threshold
        self.k_univariate = k_univariate
        self.consensus_n_splits = consensus_n_splits
        self.consensus_min_fraction = consensus_min_fraction

    def _to_dataframe(self, X):
        # Dit zorgt dat alle vervolgstappen met DataFrames en feature-namen kunnen werken.
        if isinstance(X, pd.DataFrame):
            return X.copy()

        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.feature_names_in_)

        return pd.DataFrame(X)

    def _select_features_once(self, X_df, y):
        # Dit verwijdert eerst features zonder variatie, omdat die niets bijdragen.
        constant_mask = X_df.nunique(dropna=False) <= 1
        constant_features = X_df.columns[constant_mask].to_list()
        X_non_constant = X_df.drop(columns=constant_features, errors="ignore")

        if X_non_constant.shape[1] == 0:
            return [], constant_features, []

        # Dit verwijdert vervolgens sterk gecorreleerde features om redundantie te beperken.
        corr_matrix = X_non_constant.corr(method="spearman").abs().fillna(0)
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        correlation_features = [
            col for col in upper.columns if any(upper[col] > self.corr_threshold)
        ]
        X_corr = X_non_constant.drop(columns=correlation_features, errors="ignore")

        if X_corr.shape[1] == 0:
            return [], constant_features, correlation_features

        # Dit houdt daarna de best scorende features over via univariate ANOVA-selectie.
        k = min(self.k_univariate, X_corr.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_corr, y)
        selected_features = X_corr.columns[selector.get_support()].to_list()
        return selected_features, constant_features, correlation_features

    def fit(self, X, y):
        # Dit leert op basis van de trainingsdata welke features uiteindelijk behouden blijven.
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = X_df.columns.to_list()

        y_series = pd.Series(y, index=X_df.index)
        final_features, constant_features, correlation_features = self._select_features_once(
            X_df, y_series
        )
        self.constant_features_ = constant_features
        self.correlation_features_ = correlation_features
        self.univariate_features_ = final_features.copy()

        if len(final_features) == 0:
            raise ValueError("No features left after constant, correlation, and univariate filtering.")

        # Dit controleert in meerdere interne folds welke features stabiel terugkomen.
        consensus_cv = StratifiedKFold(
            n_splits=self.consensus_n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        features_per_consensus_fold = []
        for train_idx, _ in consensus_cv.split(X_df, y_series):
            X_fold = X_df.iloc[train_idx]
            y_fold = y_series.iloc[train_idx]
            fold_features, _, _ = self._select_features_once(X_fold, y_fold)
            if fold_features:
                features_per_consensus_fold.append(fold_features)

        if not features_per_consensus_fold:
            # Dit valt terug op de eerste selectie als er geen consensusinformatie beschikbaar is.
            self.features_ = final_features.copy()
            self.consensus_feature_counts_ = Counter(final_features)
            return self

        # Dit telt hoe vaak elke feature in de consensus-folds werd geselecteerd.
        feature_counts = Counter(
            feature
            for fold_features in features_per_consensus_fold
            for feature in fold_features
        )
        min_count = max(
            1,
            int(np.ceil(self.consensus_min_fraction * len(features_per_consensus_fold))),
        )
        consensus_features = [
            feature for feature, count in feature_counts.items() if count >= min_count
        ]

        if not consensus_features:
            # Dit kiest de meest stabiele features als geen enkele feature de consensusdrempel haalt.
            consensus_features = [
                feature
                for feature, _ in feature_counts.most_common(min(len(final_features), 10))
            ]

        # Dit bewaart de uiteindelijke features in dezelfde volgorde als de eerste selectie.
        ordered_consensus = [
            feature for feature in final_features if feature in set(consensus_features)
        ]
        self.features_ = ordered_consensus if ordered_consensus else final_features.copy()
        self.consensus_feature_counts_ = feature_counts
        self.consensus_min_count_ = min_count
        self.consensus_features_per_fold_ = features_per_consensus_fold
        return self

    def transform(self, X):
        # Dit past exact de geleerde selectie toe op nieuwe data.
        X_df = self._to_dataframe(X)
        return X_df[self.features_]


def build_pipeline():
    # Dit bouwt de volledige SVM-pipeline met selectie, schaling en lineaire classificatie.
    return Pipeline(
        [
            (
                "feat_select",
                CorrUnivariateSelector(
                    corr_threshold=0.9,
                    k_univariate=20,
                    consensus_n_splits=5,
                    consensus_min_fraction=0.6,
                ),
            ),
            ("scaler", RobustScaler()),
            ("clf", SVC(kernel="linear")),
        ]
    )


def summarize_params(best_params_per_fold):
    # Dit kiest per hyperparameter de meest voorkomende beste waarde over de outer folds.
    summary = {}
    for key in best_params_per_fold[0]:
        values = [params[key] for params in best_params_per_fold]
        summary[key] = Counter(values).most_common(1)[0][0]
    return summary


def summarize_feature_stability(features_per_fold, n_folds):
    # Dit telt hoe vaak elke feature over de outer folds geselecteerd werd.
    all_selected = [feature for fold in features_per_fold for feature in fold]
    feature_counts = Counter(all_selected)

    # Dit zoekt eerst features die in alle folds voorkomen.
    consensus_features = [
        feature for feature, count in feature_counts.items() if count == n_folds
    ]
    if not consensus_features:
        # Dit versoepelt daarna naar features die in bijna alle folds voorkomen.
        consensus_features = [
            feature for feature, count in feature_counts.items() if count >= n_folds - 1
        ]

    if not consensus_features:
        # Dit valt anders terug op de meest frequent gekozen features.
        consensus_features = [
            feature
            for feature, _ in feature_counts.most_common(
                min(10, max(len(features_per_fold[0]), 1))
            )
        ]

    return feature_counts, consensus_features


def run_shap_analysis(final_pipeline, X_trainval, X_test, output_dir="shap_outputs"):
    # Dit is de oude SHAP-functie voor een finale train/test-run en blijft hier als extra helper staan.
    if shap is None:
        print("\nSHAP analysis skipped: package 'shap' is not installed.")
        print("Install first with: conda install -c conda-forge shap")
        return

    clf = final_pipeline.named_steps["clf"]
    if clf.kernel != "linear":
        print("\nSHAP analysis skipped: current final classifier is not linear.")
        return

    selector = final_pipeline.named_steps["feat_select"]
    scaler = final_pipeline.named_steps["scaler"]
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # SHAP moet dezelfde geselecteerde features zien als de classifier.
    train_selected = pd.DataFrame(
        selector.transform(X_trainval),
        columns=selector.features_,
        index=X_trainval.index,
    )
    test_selected = pd.DataFrame(
        selector.transform(X_test),
        columns=selector.features_,
        index=X_test.index,
    )

    # We gebruiken ook dezelfde scaling als in de getrainde pipeline.
    train_scaled = pd.DataFrame(
        scaler.transform(train_selected),
        columns=selector.features_,
        index=X_trainval.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_selected),
        columns=selector.features_,
        index=X_test.index,
    )

    # Kleine achtergrondset houdt de SHAP-berekening sneller en stabiel.
    background_size = min(100, len(train_scaled))
    background = train_scaled.sample(background_size, random_state=RANDOM_STATE)

    explainer = shap.LinearExplainer(clf, background)
    shap_values = explainer(test_scaled)

    importance_df = (
        pd.DataFrame(
            {
                "feature": test_scaled.columns,
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 40)
    print("SHAP mean absolute importance")
    print(importance_df.to_string(index=False))

    shap.plots.beeswarm(shap_values, max_display=len(test_scaled.columns), show=False)
    plt.tight_layout()
    plt.savefig(output_path / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.plots.bar(shap_values, max_display=len(test_scaled.columns), show=False)
    plt.tight_layout()
    plt.savefig(output_path / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP plots to: {output_path.resolve()}")


def run_nested_shap_analysis(nested_shap_payload, output_dir="shap_outputs_nested_cv_svm"):
    # Sla SHAP over als de dependency ontbreekt.
    if shap is None:
        print("\nNested SHAP analysis skipped: package 'shap' is not installed.")
        return

    # Zonder fold-resultaten kunnen we geen nested SHAP-samenvatting maken.
    if not nested_shap_payload:
        print("\nNested SHAP analysis skipped: no SHAP payload available.")
        return

    # Neem de unie van alle features over alle outer folds, zodat we fold-resultaten
    # later in een gezamenlijke tabel kunnen uitlijnen.
    all_features = sorted(
        {
            feature
            for fold_payload in nested_shap_payload
            for feature in fold_payload["feature_names"]
        }
    )
    if not all_features:
        print("\nNested SHAP analysis skipped: no features available.")
        return

    # Maak de outputmap aan voor de samenvatting.
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    combined_shap_frames = []

    for fold_payload in nested_shap_payload:
        feature_names = fold_payload["feature_names"]
        # Vul ontbrekende features per fold met 0 zodat alle folds dezelfde kolommen hebben.
        shap_frame = pd.DataFrame(
            fold_payload["shap_values"],
            columns=feature_names,
            index=fold_payload["index"],
        ).reindex(columns=all_features, fill_value=0.0)

        combined_shap_frames.append(shap_frame)

    # Voeg SHAP-waarden van alle outer folds samen voor een nested CV-overzicht.
    combined_shap = pd.concat(combined_shap_frames, axis=0).sort_index()

    # Bereken per feature de gemiddelde absolute SHAP-waarde als globale belangrijkheid.
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

    # Bewaar alleen de compacte SHAP-samenvatting als csv.
    importance_df.to_csv(output_path / "nested_shap_importance.csv", index=False)

    print("\n" + "=" * 40)
    print("Nested CV SHAP mean absolute importance")
    print(importance_df.to_string(index=False))
    print(f"Saved nested SHAP summary to: {output_path.resolve()}")


def save_roc_curve(
    y_true,
    y_scores,
    roc_auc,
    output_path="roc_curve_final_test.png",
    title="ROC Curve",
):
    # Dit berekent een ROC-curve en slaat die op als png-bestand.
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve to: {Path(output_path).resolve()}")


def run_nested_cv(X_trainval, y_trainval):
    # Dit maakt een outer en inner cross-validation voor eerlijke nested modelselectie.
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Dit definieert welke hyperparameters we tijdens de nested search gaan testen.
    param_dist = {
        "feat_select__corr_threshold": [0.8, 0.85, 0.9],
        "feat_select__k_univariate": [10, 15],
        "feat_select__consensus_min_fraction": [0.6, 0.8],
        "clf__C": [0.01, 0.1, 1, 10],
    }

    # Dit verzamelt alle fold-uitkomsten zodat we nested prestaties achteraf kunnen samenvatten.
    all_y_outer = []
    all_scores_outer = []
    features_per_fold = []
    best_params_per_fold = []
    all_fold_scores = []
    all_fold_y = []
    fold_aucs = []
    # Hier bewaren we per outer fold de SHAP-resultaten van de trainingsdata.
    nested_shap_payload = []

    # Dit maakt een scorer waarmee de hyperparametersearch op F2 wordt afgestemd.
    f2_score = make_scorer(fbeta_score, beta=2)

    print("Start nested cross-validation...")

    for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(
        outer_cv.split(X_trainval, y_trainval), start=1
    ):
        # Dit splitst per outer fold in een trainings- en validatiedeel.
        X_outer_train = X_trainval.iloc[outer_train_idx].copy()
        X_outer_val = X_trainval.iloc[outer_val_idx].copy()
        y_outer_train = y_trainval.iloc[outer_train_idx]
        y_outer_val = y_trainval.iloc[outer_val_idx]

        # Dit zoekt binnen de inner CV naar de beste hyperparameters op basis van F2.
        search = RandomizedSearchCV(
            estimator=build_pipeline(),
            param_distributions=param_dist,
            n_iter=20,
            cv=inner_cv,
            scoring=f2_score,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        search.fit(X_outer_train, y_outer_train)

        # Dit pakt het beste model en leest de geselecteerde features uit.
        best_model = search.best_estimator_
        selector = best_model.named_steps["feat_select"]

        best_params_per_fold.append(search.best_params_)
        features_per_fold.append(selector.features_)

        scores_outer = best_model.decision_function(X_outer_val)
        all_y_outer.extend(y_outer_val)
        all_scores_outer.extend(scores_outer)

        # Bewaar outer-fold scores en labels zodat we achteraf nested F2 kunnen berekenen.
        all_fold_scores.append(scores_outer)
        all_fold_y.append(y_outer_val)

        # Bereken de ROC-AUC van deze outer fold afzonderlijk.
        fold_auc = roc_auc_score(y_outer_val, scores_outer)
        fold_aucs.append(fold_auc)

        # Bereken SHAP op de trainingsdata van deze outer fold om modelgedrag binnen
        # nested CV te interpreteren zonder de testset te gebruiken.
        if shap is not None and best_model.named_steps["clf"].kernel == "linear":
            selector = best_model.named_steps["feat_select"]
            scaler = best_model.named_steps["scaler"]

            # Gebruik exact dezelfde geselecteerde en geschaalde features als de classifier ziet.
            train_selected = pd.DataFrame(
                selector.transform(X_outer_train),
                columns=selector.features_,
                index=X_outer_train.index,
            )
            shap_selected = pd.DataFrame(
                selector.transform(X_outer_train),
                columns=selector.features_,
                index=X_outer_train.index,
            )
            train_scaled = pd.DataFrame(
                scaler.transform(train_selected),
                columns=selector.features_,
                index=X_outer_train.index,
            )
            shap_scaled = pd.DataFrame(
                scaler.transform(shap_selected),
                columns=selector.features_,
                index=X_outer_train.index,
            )

            # Neem een kleine achtergrondset voor een stabiele en snellere SHAP-berekening.
            background_size = min(100, len(train_scaled))
            background = train_scaled.sample(background_size, random_state=RANDOM_STATE)
            explainer = shap.LinearExplainer(best_model.named_steps["clf"], background)
            shap_values = explainer(shap_scaled)

            # Bewaar per fold alleen wat nodig is voor de uiteindelijke nested samenvatting.
            nested_shap_payload.append(
                {
                    "feature_names": selector.features_,
                    "shap_values": shap_values.values,
                    "index": shap_scaled.index,
                }
            )

    # Bereken de totale nested ROC-AUC over alle outer-fold voorspellingen samen.
    nested_auc = roc_auc_score(all_y_outer, all_scores_outer)

    # Zet decision scores om naar klasselabels met de standaardgrens 0 voor lineaire SVM.
    pred_labels = np.concatenate([(s > 0).astype(int) for s in all_fold_scores])
    true_labels = np.concatenate([y.values for y in all_fold_y])
    nested_f2 = fbeta_score(true_labels, pred_labels, beta=2)

    feature_counts, consensus_features = summarize_feature_stability(
        features_per_fold, outer_cv.get_n_splits()
    )
    final_params = summarize_params(best_params_per_fold)

    return (
        nested_auc,
        nested_f2,
        fold_aucs,
        feature_counts,
        consensus_features,
        final_params,
        np.array(all_y_outer),
        np.array(all_scores_outer),
        nested_shap_payload,
    )
def main():
    # Dit laadt de data en houdt alleen numerieke features plus het label over.
    data = load_data()
    X = data.select_dtypes(include=[np.number]).copy()
    y = data["label"].map({"benign": 0, "malignant": 1}).astype(int)

    # Dit maakt alvast een vaste train/test-split, terwijl nested CV alleen op trainval draait.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Dit voert nested CV uit om prestaties, stabiele features en beste hyperparameters te schatten.
    (
        nested_auc,
        nested_f2,
        fold_aucs,
        feature_counts,
        consensus_features,
        final_params,
        all_y_outer,
        all_scores_outer,
        nested_shap_payload,
    ) = run_nested_cv(X_trainval, y_trainval)


    print("\n" + "=" * 40)
    print("Feature stability")
    print(f"Unique selected features: {len(feature_counts)}")
    print(f"Consensus/stable features: {consensus_features}")

    print("\n" + "=" * 40)
    print(f"Nested CV ROC-AUC: {nested_auc:.3f}")
    print(f"Nested CV F2-score: {nested_f2:.3f}")
    print(f"Most common best params over outer folds: {final_params}")

    # Dit bouwt de finale pipeline op met de meest gekozen hyperparameters.
    final_pipeline = build_pipeline()
    final_pipeline.set_params(**final_params)

    # Dit voert nog een gewone 5-fold CV uit met de gekozen hyperparameters.
    regular_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    regular_cv_scores = cross_val_score(
        final_pipeline,
        X_trainval,
        y_trainval,
        cv=regular_cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    print("\n" + "=" * 40)
    print(
        f"5-fold CV ROC-AUC: {regular_cv_scores.mean():.3f} "
        f"+/- {regular_cv_scores.std():.3f}"
    )

    # Bewaar een ROC-curve op basis van alle out-of-fold nested CV-voorspellingen.
    save_roc_curve(
        all_y_outer,
        all_scores_outer,
        nested_auc,
        output_path="roc_curve_nested_cv_svm_uni.png",
        title="Nested CV ROC Curve (SVM Univariate)",
    )
    # Maak daarnaast een compacte SHAP-samenvatting over alle outer folds.
    run_nested_shap_analysis(nested_shap_payload)




# %%
def SVM_Uni(test=False):
    # Dit is de functie die Final.py gebruikt om SVM-resultaten op te halen.
    data = load_data()
    X = data.select_dtypes(include=[np.number]).copy()
    y = data["label"].map({"benign": 0, "malignant": 1}).astype(int)

    # Dit maakt dezelfde vaste train/test-split als in main().
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Dit voert nested CV uit op alleen het train/validation-deel.
    (
        nested_auc,
        nested_f2,
        fold_aucs,
        f_counts,
        c_features,
        f_params,
        all_y_outer,
        all_scores_outer,
        nested_shap_payload,
    ) = run_nested_cv(X_trainval, y_trainval)
    # Dit zet de kernresultaten klaar die Final.py gebruikt voor modelvergelijking.
    results = {
        "nested_auc_SVM_uni": float(nested_auc),
        "nested_f2_SVM_uni": float(nested_f2),
        "fold_aucs_SVM_uni": fold_aucs,
    }

    # Maak bij elke run de nested ROC-curve en SHAP-samenvatting aan.
    save_roc_curve(
        all_y_outer,
        all_scores_outer,
        nested_auc,
        output_path="roc_curve_nested_cv_svm_uni.png",
        title="Nested CV ROC Curve (SVM Univariate)",
    )
    run_nested_shap_analysis(nested_shap_payload)

    if test:
        # Train pas nu het finale model op alle train/validation-data voor de echte testset.
        final_pipeline = build_pipeline()
        final_pipeline.set_params(**f_params)
        final_pipeline.fit(X_trainval, y_trainval)
        test_scores = final_pipeline.decision_function(X_test)
        test_auc = roc_auc_score(y_test, test_scores)
        test_pred = (test_scores > 0).astype(int)
        test_f2 = fbeta_score(y_test, test_pred, beta=2)
        # Geef ook de ruwe testscores en labels terug zodat Final.py zelf de ROC-curve kan maken.
        results["test_auc"] = float(test_auc)
        results["test_f2"] = float(test_f2)
        results["test_scores"] = test_scores
        results["y_test"] = y_test.to_numpy()

    return results

# %%
if __name__ == "__main__":
    # Dit zorgt dat main() alleen draait als je dit bestand direct uitvoert.
    main()
