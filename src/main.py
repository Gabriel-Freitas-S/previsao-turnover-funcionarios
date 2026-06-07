"""
PREVISAO DE ROTATIVIDADE DE FUNCIONARIOS (TURNOVER)
====================================================
Projeto de Machine Learning - Classificacao Binaria

Problema:
    Prever se um funcionario tem alta probabilidade de deixar a empresa,
    com base em atributos de satisfacao, desempenho e carga de trabalho.

Abordagem:
    - Regressao Logistica (modelo linear, interpretavel)
    - Random Forest (ensemble nao-linear, robusto)
    - Otimizacao via GridSearchCV com validacao cruzada estratificada
    - Metrica principal: F1-Score (devido ao desbalanceamento das classes)

Dataset: HR Analytics - Kaggle (Giri Pujar)
    14.999 registros, 10 atributos.
    Colunas (traduzidas para PT-BR):
        nivel_satisfacao, ultima_avaliacao, numero_projetos,
        media_horas_mensais, tempo_empresa, acidente_trabalho,
        saiu (target), promocao_ultimos_5anos, departamento, salario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HR_Analytics.csv")
RANDOM_STATE = 42


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def explore_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Shape: {df.shape}")
    print(f"\nColunas: {df.columns.tolist()}")
    print(f"\nTipos:\n{df.dtypes.value_counts()}")
    print(f"\nValores nulos:\n{df.isnull().sum().sum()}")
    target_dist = df["saiu"].value_counts()
    target_pct = df["saiu"].value_counts(normalize=True) * 100
    print(f"\nDistribuicao 'saiu':\n{target_dist}\n{target_pct.round(2)}")
    return df


def plot_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Analise Exploratoria de Dados (EDA) — HR Analytics", fontsize=16)

    df_plot = df.copy()
    df_plot["Rotatividade"] = df_plot["saiu"].map({0: "Nao", 1: "Sim"})

    # 1. Distribuição de Turnover
    sns.countplot(data=df_plot, x="Rotatividade", hue="Rotatividade",
                  ax=axes[0, 0], legend=False, palette=["#4CAF50", "#F44336"])
    axes[0, 0].set_title("Distribuicao de Turnover")
    axes[0, 0].set_xlabel("")

    # 2. Nivel de Satisfacao
    sns.boxplot(data=df_plot, x="Rotatividade", y="nivel_satisfacao", hue="Rotatividade",
                ax=axes[0, 1], legend=False, palette=["#4CAF50", "#F44336"])
    axes[0, 1].set_title("Satisfacao vs Turnover")
    axes[0, 1].set_ylabel("Nivel de Satisfacao")

    # 3. Media de Horas Mensais
    sns.boxplot(data=df_plot, x="Rotatividade", y="media_horas_mensais", hue="Rotatividade",
                ax=axes[0, 2], legend=False, palette=["#4CAF50", "#F44336"])
    axes[0, 2].set_title("Horas Mensais vs Turnover")
    axes[0, 2].set_ylabel("Media de Horas Mensais")

    # 4. Turnover por Departamento
    att_by_dept = df_plot.groupby("departamento")["saiu"].value_counts(normalize=True).unstack() * 100
    if 1 in att_by_dept.columns:
        att_by_dept = att_by_dept[[1, 0]] if 0 in att_by_dept.columns else att_by_dept
    else:
        att_by_dept[1] = 0
        att_by_dept[0] = 100 - att_by_dept[1]
    att_by_dept = att_by_dept.sort_values(1, ascending=False)
    att_by_dept.plot(kind="bar", stacked=True, ax=axes[1, 0], color=["#F44336", "#4CAF50"])
    axes[1, 0].set_title("Turnover por Departamento (%)")
    axes[1, 0].set_ylabel("Percentual")
    axes[1, 0].legend(["Saiu", "Ficou"], title="Rotatividade")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 5. Turnover por Faixa Salarial
    sal_order = ["baixo", "medio", "alto"]
    sns.countplot(data=df_plot, x="salario", hue="Rotatividade",
                  ax=axes[1, 1], order=sal_order, palette=["#4CAF50", "#F44336"])
    axes[1, 1].set_title("Salario vs Turnover")
    axes[1, 1].set_xlabel("Faixa Salarial")

    # 6. Ultima Avaliacao
    sns.boxplot(data=df_plot, x="Rotatividade", y="ultima_avaliacao", hue="Rotatividade",
                ax=axes[1, 2], legend=False, palette=["#4CAF50", "#F44336"])
    axes[1, 2].set_title("Ultima Avaliacao vs Turnover")
    axes[1, 2].set_ylabel("Pontuacao da Ultima Avaliacao")

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, "slides", "eda_plots.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Graficos EDA salvos em {output_path}")


def preprocess_data(df: pd.DataFrame):
    df_proc = df.copy()

    # Variavel alvo
    y = df_proc["saiu"].astype(int)

    # Features — remover somente o target
    cols_to_drop = ["saiu"]
    X = df_proc.drop(columns=[c for c in cols_to_drop if c in df_proc.columns], errors="ignore")

    # Limpar strings de colunas texto
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].str.strip()

    return X, y


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"\nFeatures numericas ({len(numeric_features)}): {numeric_features}")
    print(f"Features categoricas ({len(categorical_features)}): {categorical_features}")

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return preprocessor, numeric_features, categorical_features


def create_models():
    models = {
        "RegressaoLogistica": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }
    param_grids = {
        "RegressaoLogistica": {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
        },
        "RandomForest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
        },
        "GradientBoosting": {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__max_depth": [3, 5],
        },
    }
    return models, param_grids


def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor):
    models, param_grids = create_models()
    results = []

    for name, model in models.items():
        print(f"\n>>> Treinando {name} com GridSearchCV...")

        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        gs = GridSearchCV(
            pipe, param_grids[name],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1",
            n_jobs=-1,
            verbose=1
        )
        gs.fit(X_train, y_train)

        y_val_pred = gs.predict(X_val)
        y_val_proba = gs.predict_proba(X_val)[:, 1]

        y_test_pred = gs.predict(X_test)
        y_test_proba = gs.predict_proba(X_test)[:, 1]

        classifier = gs.best_estimator_.named_steps["classifier"]
        feature_importances = None
        if hasattr(classifier, "feature_importances_"):
            feature_importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            feature_importances = classifier.coef_[0]

        feature_names = gs.best_estimator_.named_steps["preprocessor"].get_feature_names_out()

        results.append({
            "model": name,
            "best_estimator": gs.best_estimator_,
            "best_params": gs.best_params_,
            "best_score": gs.best_score_,

            "val_f1": f1_score(y_val, y_val_pred),
            "val_recall": recall_score(y_val, y_val_pred),
            "val_roc_auc": roc_auc_score(y_val, y_val_proba),

            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "f1_score": f1_score(y_test, y_test_pred),
            "roc_auc": roc_auc_score(y_test, y_test_proba),
            "y_pred": y_test_pred,
            "y_proba": y_test_proba,
            "confusion_matrix": confusion_matrix(y_test, y_test_pred),
            "classification_report": classification_report(y_test, y_test_pred),
            "feature_importances": feature_importances,
            "feature_names": feature_names
        })
    return results


def plot_results(results: list, y_test: pd.Series) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Comparacao de Modelos — Previsao de Turnover (HR Analytics)", fontsize=16, fontweight="bold", y=0.98)

    metrics_df = pd.DataFrame([
        {"Modelo": r["model"], "Acuracia": r["accuracy"],
         "Precisao": r["precision"], "Recall": r["recall"],
         "F1-Score": r["f1_score"], "ROC-AUC": r["roc_auc"]}
        for r in results
    ]).set_index("Modelo")

    metrics_df.plot(kind="bar", ax=axes[0, 0], rot=0, colormap="viridis")
    axes[0, 0].set_title("Metricas de Desempenho", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].set_xlabel("")

    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        axes[0, 1].plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0, 1].set_title("Curva ROC", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Taxa de Falso Positivo")
    axes[0, 1].set_ylabel("Taxa de Verdadeiro Positivo")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    fi_data = []
    feature_names = None
    for r in results:
        if r["model"] == "RandomForest":
            fi = r.get("feature_importances")
            feature_names = r.get("feature_names")
            if fi is not None:
                fi_data.append((r["model"], fi))

    if fi_data:
        _, fi = fi_data[0]
        n_top = min(10, len(fi))
        fi_sorted = fi.argsort()[-n_top:][::-1]
        if feature_names is not None:
            cleaned_names = [str(feature_names[idx]).replace("cat__", "").replace("num__", "") for idx in fi_sorted]
            axes[0, 2].barh(range(n_top), fi[fi_sorted], color="steelblue")
            axes[0, 2].set_yticks(range(n_top))
            axes[0, 2].set_yticklabels(cleaned_names, fontsize=8)
        else:
            axes[0, 2].barh(range(n_top), fi[fi_sorted], color="steelblue")
            axes[0, 2].set_yticks(range(n_top))
            axes[0, 2].set_yticklabels(range(n_top))
        axes[0, 2].set_title(f"Top {n_top} Atributos — Random Forest", fontsize=12, fontweight="bold")
        axes[0, 2].invert_yaxis()
    else:
        axes[0, 2].text(0.5, 0.5, "Importancia de atributos\nindisponivel", ha="center", va="center")
        axes[0, 2].set_title("Importancia dos Atributos", fontsize=12, fontweight="bold")

    for i, r in enumerate(results):
        ax = axes[1, i]
        sns.heatmap(r["confusion_matrix"], annot=True, fmt="d",
                    cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"Matriz de Confusao\n{r['model']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        ax.set_xticklabels(["Ficou", "Saiu"])
        ax.set_yticklabels(["Ficou", "Saiu"])

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, "slides", "model_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Grafico salvo em {output_path}")


def main():
    print("=" * 64)
    print("  PREVISAO DE ROTATIVIDADE DE FUNCIONARIOS")
    print("  HR Analytics — Kaggle (Giri Pujar)")
    print("  Machine Learning — Classificacao Binaria (Turnover)")
    print("=" * 64)

    df = load_data()

    explore_data(df)

    plot_eda(df)

    X, y = preprocess_data(df)

    preprocessor, num_feat, cat_feat = build_preprocessor(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"\nDivisao (60/20/20): Treino={X_train.shape[0]}, Validacao={X_val.shape[0]}, Teste={X_test.shape[0]}")
    print(f"Turnover em treino:\n{y_train.value_counts(normalize=True).mul(100).round(2)}")

    results = train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)

    print("\n" + "=" * 64)
    print("RESULTADOS — CLASSIFICACAO BINARIA (TURNOVER)")
    print("=" * 64)

    best_model_name = None
    best_val_f1 = -1.0
    best_pipeline = None

    for r in results:
        print(f"\n{'='*40}")
        print(f"Modelo: {r['model']}")
        print(f"Melhores hiperparametros: {r['best_params']}")
        print(f"Melhor F1 (CV Treino): {r['best_score']:.4f}")
        print(f"--- VALIDACAO ---")
        print(f"F1-Score:  {r['val_f1']:.4f}")
        print(f"Recall:    {r['val_recall']:.4f}")
        print(f"ROC-AUC:   {r['val_roc_auc']:.4f}")
        print(f"--- TESTE FINAL ---")
        print(f"Acuracia:  {r['accuracy']:.4f}")
        print(f"Precisao:  {r['precision']:.4f}")
        print(f"Recall:    {r['recall']:.4f}")
        print(f"F1-Score:  {r['f1_score']:.4f}")
        print(f"ROC-AUC:   {r['roc_auc']:.4f}")
        print(f"\nRelatorio Teste:\n{r['classification_report']}")
        print(f"Matriz de Confusao Teste:\n{r['confusion_matrix']}")

        if r['val_f1'] > best_val_f1:
            best_val_f1 = r['val_f1']
            best_model_name = r['model']
            best_pipeline = r['best_estimator']

    print(f"\n>>> Melhor modelo selecionado (Val F1={best_val_f1:.4f}): {best_model_name}")

    model_path = os.path.join(BASE_DIR, "modelo_turnover.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"Melhor modelo persistido em: {model_path}")

    metrics_dict = {}
    for r in results:
        metrics_dict[r["model"]] = {
            "accuracy": float(r["accuracy"]),
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "f1_score": float(r["f1_score"]),
            "roc_auc": float(r["roc_auc"])
        }
    metrics_dict["best_model"] = best_model_name

    metrics_path = os.path.join(BASE_DIR, "slides", "metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metricas exportadas para: {metrics_path}")

    plot_results(results, y_test)
    print("\nProjeto concluido! Verifique os graficos em slides/")


if __name__ == "__main__":
    main()
