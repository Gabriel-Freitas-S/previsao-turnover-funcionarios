"""
PREVISAO DE ROTATIVIDADE DE FUNCIONARIOS (TURNOVER)
====================================================
Projeto de Machine Learning - Classificacao Binaria

Problema:
    Prever se um funcionario tem alta probabilidade de deixar a empresa,
    com base em atributos demograficos, profissionais e de satisfacao.

Abordagem:
    - Regressao Logistica (modelo linear, interpretavel)
    - Random Forest (ensemble nao-linear, robusto)
    - Otimizacao via GridSearchCV com validacao cruzada estratificada
    - Metrica principal: F1-Score (devido ao desbalanceamento 84/16)

Dataset: IBM HR Analytics Employee Attrition & Performance (Kaggle)
    1470 registros, 35 atributos, 0 valores nulos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# Suprime warnings de convergencia e deprecacao para manter a saida limpa
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Caminho relativo ao diretorio src/ - os graficos serao salvos em slides/
DATA_PATH = "../data/HR_Employee_Attrition.csv"
RANDOM_STATE = 42

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Carrega o dataset IBM HR a partir de um arquivo CSV.

    O encoding utf-8-sig remove o BOM (Byte Order Mark) que alguns
    arquivos CSV exportados do Excel podem conter no inicio.

    Args:
        path: Caminho absoluto ou relativo para o arquivo CSV.

    Returns:
        DataFrame do Pandas com os dados carregados.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df

def explore_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exibe informacoes basicas sobre o dataset: shape, colunas, tipos,
    valores nulos e distribuicao da variavel alvo Attrition.

    Args:
        df: DataFrame a ser explorado.

    Returns:
        O mesmo DataFrame (para encadeamento de operacoes).
    """
    print(f"Shape: {df.shape}")
    print(f"\nColunas: {df.columns.tolist()}")
    print(f"\nTipos:\n{df.dtypes.value_counts()}")
    print(f"\nValores nulos:\n{df.isnull().sum().sum()}")
    attrition_dist = df["Attrition"].value_counts()
    attrition_pct = df["Attrition"].value_counts(normalize=True) * 100
    print(f"\nDistribuicao Attrition:\n{attrition_dist}\n{attrition_pct.round(2)}")
    return df

def plot_eda(df: pd.DataFrame) -> None:
    """
    Gera graficos de Analise Exploratoria de Dados (EDA) e salva como PNG.

    6 subplots abordando:
        - Distribuicao da variavel alvo (countplot)
        - Idade vs Attrition (boxplot) - funcionarios mais jovens saem mais?
        - Renda mensal vs Attrition (boxplot) - menor renda leva a turnover?
        - Attrition por departamento (barras empilhadas)
        - Hora extra vs Attrition (countplot)
        - Attrition por cargo (countplot)

    Os graficos sao salvos em slides/eda_plots.png para uso nos slides.

    Args:
        df: DataFrame com os dados completos.
    """
    # Cria uma figura 2x3 com tamanho 15x10 polegadas
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Analise Exploratoria de Dados", fontsize=16)

    # (1,1) - Distribuicao da variavel alvo (balanceamento das classes)
    sns.countplot(data=df, x="Attrition", hue="Attrition", ax=axes[0, 0], legend=False)
    axes[0, 0].set_title("Distribuicao de Attrition")

    # (1,2) - Idade vs Attrition: funcionarios mais jovens tendem a sair mais
    sns.boxplot(data=df, x="Attrition", y="Age", hue="Attrition", ax=axes[0, 1], legend=False)
    axes[0, 1].set_title("Idade vs Attrition")

    # (1,3) - Renda mensal: turnover e mais frequente em salarios mais baixos
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", hue="Attrition", ax=axes[0, 2], legend=False)
    axes[0, 2].set_title("Renda Mensal vs Attrition")

    # (2,1) - Proporcao de attrition por departamento (barras empilhadas)
    att_by_dept = df.groupby("Department")["Attrition"].value_counts(normalize=True).unstack() * 100
    att_by_dept.plot(kind="bar", stacked=True, ax=axes[1, 0], color=["#4CAF50", "#F44336"])
    axes[1, 0].set_title("Attrition por Departamento (%)")
    axes[1, 0].set_ylabel("Percentual")
    axes[1, 0].legend(title="Attrition")

    # (2,2) - Horas extras aumentam significativamente o risco de turnover
    sns.countplot(data=df, x="OverTime", hue="Attrition", ax=axes[1, 1])
    axes[1, 1].set_title("Hora Extra vs Attrition")

    # (2,3) - Distribuicao de attrition por cargo (rotacionar labels para legibilidade)
    sns.countplot(data=df, x="JobRole", hue="Attrition", ax=axes[1, 2])
    axes[1, 2].set_title("Attrition por Cargo")
    axes[1, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("../slides/eda_plots.png", dpi=150, bbox_inches="tight")
    print("Graficos EDA salvos em slides/eda_plots.png")

def preprocess_data(df: pd.DataFrame):
    """
    Separa features (X) e target (y) e remove colunas irrelevantes.

    Colunas removidas:
        - Attrition: variavel alvo (vira y)
        - EmployeeCount: valor constante (1 para todos os registros)
        - StandardHours: valor constante (80 para todos)
        - Over18: valor constante (Y para todos)
        - EmployeeNumber: identificador unico, sem poder preditivo

    A variavel alvo e mapeada de categorico (Yes/No) para binario (1/0).

    Args:
        df: DataFrame completo.

    Returns:
        X (DataFrame): features para modelagem.
        y (Series): variavel alvo binaria (1=Yes, 0=No).
    """
    # EmployeeCount, StandardHours e Over18 possuem variancia zero (valor unico
    # para todas as linhas) - nao agregam informacao ao modelo.
    # EmployeeNumber e um identificador sequencial, sem correlacao com turnover.
    X = df.drop(columns=["Attrition", "EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"])
    y = df["Attrition"].map({"Yes": 1, "No": 0})
    return X, y

def build_preprocessor(X: pd.DataFrame):
    """
    Constroi um ColumnTransformer para pre-processamento diferenciado.

    Variaveis numericas (int64/float64):
        - Padronizacao com StandardScaler (media=0, desvio=1)
        - Necessario para modelos lineares (Regressao Logistica) que sao
          sensiveis a escalas diferentes entre features.

    Variaveis categoricas (object):
        - OneHotEncoder com drop='first' para evitar multicolinearidade
          (k-1 categorias instead of k)
        - handle_unknown='ignore' para tolerar categorias nao vistas
          em teste (robustez).

    Args:
        X: DataFrame de features (antes da divisao treino/teste).

    Returns:
        preprocessor (ColumnTransformer): pipeline de pre-processamento.
        numeric_features (list): nomes das colunas numericas identificadas.
        categorical_features (list): nomes das colunas categoricas identificadas.
    """
    # Identificacao automatica dos tipos de feature
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"\nFeatures numericas ({len(numeric_features)}): {numeric_features}")
    print(f"Features categoricas ({len(categorical_features)}): {categorical_features}")

    # Pipeline individual para cada tipo
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    # Combinacao dos pipelines em um unico transformador
    # Isso garante que o fit seja feito apenas no conjunto de treino,
    # evitando data leakage (vazamento de informacao do teste para o treino).
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return preprocessor, numeric_features, categorical_features

def create_models():
    """
    Define os modelos e os grids de hiperparametros para o GridSearchCV.

    Modelo 1 - Regressao Logistica:
        - class_weight='balanced': ajusta pesos inversamente proporcionais
          as frequencias das classes. Como temos 84% No e 16% Yes, a classe
          minoritaria recebe peso maior para evitar que o modelo ignore os
          casos de turnover.
        - max_iter=1000: garantia de convergencia para o solver lbfgs.

    Modelo 2 - Random Forest:
        - class_weight='balanced': mesmo principio de balanceamento.
        - random_state=42: reprodutibilidade.

    Grids:
        - Reg. Logistica: busca o melhor C (regularizacao inversa) para
          controlar overfitting. Valores menores = mais regularizacao.
        - Random Forest: busca profundidade maxima, numero de arvores,
          e criterios de parada (min_samples_split, min_samples_leaf).

    Returns:
        models (dict): nome -> objeto estimator.
        param_grids (dict): nome -> dict de hiperparametros para buscar.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    param_grids = {
        "LogisticRegression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
        },
        "RandomForest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
        },
    }
    return models, param_grids

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    """
    Treina cada modelo com GridSearchCV e avalia no conjunto de teste.

    O pipeline completo e:
        1. Pre-processamento (fit apenas no treino)
        2. Classificador (treinado com os dados pre-processados)

    O GridSearchCV utiliza:
        - StratifiedKFold (5 folds): preserva a proporcao das classes
          em cada fold para evitar folds sem a classe minoritaria.
        - scoring='f1': otimiza o F1-Score da classe positiva (1),
          que e a metrica mais adequada para dados desbalanceados.
        - n_jobs=-1: utiliza todos os nucleos da CPU.

    Args:
        X_train: features de treino.
        X_test: features de teste.
        y_train: rotulos de treino.
        y_test: rotulos de teste.
        preprocessor: ColumnTransformer ja configurado.

    Returns:
        results (list): lista de dicionarios com metricas de cada modelo.
    """
    models, param_grids = create_models()
    results = []

    for name, model in models.items():
        print(f"\n>>> Treinando {name} com GridSearchCV...")

        # Pipeline: pre-processamento -> classificador
        # Isso garante que o ColumnTransformer seja aplicado corretamente
        # dentro de cada fold da validacao cruzada, sem data leakage.
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        # Configuracao da busca em grade com validacao cruzada
        gs = GridSearchCV(
            pipe, param_grids[name],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1",
            n_jobs=-1,
            verbose=1
        )
        gs.fit(X_train, y_train)

        # Predicoes no conjunto de teste
        y_pred = gs.predict(X_test)
        # Probabilidades da classe positiva (indice 1) para ROC-AUC
        y_proba = gs.predict_proba(X_test)[:, 1]

        # Colecao de todas as metricas para comparacao posterior
        results.append({
            "model": name,
            "best_params": gs.best_params_,
            "best_score": gs.best_score_,          # F1 medio da validacao cruzada
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "y_pred": y_pred,
            "y_proba": y_proba,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
        })
    return results

def plot_results(results: list, y_test: pd.Series) -> None:
    """
    Gera grafico comparativo entre os modelos e salva como PNG.

    Layout 2x2:
        (1,1) - Barras comparativas: Acuracia, Precisao, Recall, F1, AUC
        (1,2) - Curva ROC sobreposta dos dois modelos
        (2,1) - Matrizes de confusao (sobrepoe a ultima)
        (2,2) - Top 10 features mais importantes (Random Forest)

    Args:
        results: lista de dicionarios com as metricas de cada modelo.
        y_test: rotulos verdadeiros do conjunto de teste.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparacao de Modelos - Previsao de Turnover", fontsize=16)

    # (1,1) - Grafico de barras com todas as metricas lado a lado
    metrics_df = pd.DataFrame([
        {"Model": r["model"], "Acuracia": r["accuracy"],
         "Precisao": r["precision"], "Recall": r["recall"],
         "F1-Score": r["f1_score"], "ROC-AUC": r["roc_auc"]}
        for r in results
    ]).set_index("Model")

    metrics_df.plot(kind="bar", ax=axes[0, 0], rot=0, colormap="viridis")
    axes[0, 0].set_title("Metricas de Desempenho")
    axes[0, 0].set_ylim(0, 1)      # Metricas sao proporcoes entre 0 e 1
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(axis="y", alpha=0.3)

    # (2,1) - Matriz de confusao (a ultima sobrescreve, mas valores sao exibidos)
    # Idealmente seria um subplot para cada modelo, mas para simplificar
    # exibimos a matriz do ultimo modelo processado.
    for r in results:
        sns.heatmap(r["confusion_matrix"], annot=True, fmt="d",
                    cmap="Blues", ax=axes[1, 0])
        axes[1, 0].set_title(f"Matriz de Confusao - {r['model']}")
        axes[1, 0].set_xlabel("Previsto")
        axes[1, 0].set_ylabel("Real")

    # (1,2) - Curva ROC: taxa de verdadeiro positivo vs taxa de falso positivo
    # AUC (Area Under the Curve) mede a capacidade de discriminacao do modelo.
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        axes[0, 1].plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.3f})")
    # Linha diagonal representa um classificador aleatorio (AUC=0.5)
    axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0, 1].set_title("Curva ROC")
    axes[0, 1].set_xlabel("Taxa de Falso Positivo")
    axes[0, 1].set_ylabel("Taxa de Verdadeiro Positivo")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # (2,2) - Importancia de features do Random Forest (top 10)
    # O Random Forest possui o atributo feature_importances_ que indica
    # o peso relativo de cada feature na decisao das arvores.
    fi_data = []
    for r in results:
        if r["model"] == "RandomForest":
            fi = r.get("feature_importances")
            if fi is not None:
                fi_data.append((r["model"], fi))

    if fi_data:
        _, fi = fi_data[0]
        # Indices das 10 features mais importantes, ordenados decrescentemente
        fi_sorted = fi.argsort()[-10:][::-1]
        axes[1, 1].barh(range(10), fi[fi_sorted], color="steelblue")
        axes[1, 1].set_title("Top 10 Features - Random Forest")
        axes[1, 1].set_yticks(range(10))
        axes[1, 1].invert_yaxis()  # Maior importancia no topo
    else:
        axes[1, 1].text(0.5, 0.5, "Feature importance\nindisponivel",
                        ha="center", va="center")
        axes[1, 1].set_title("Importancia das Features")

    plt.tight_layout()
    plt.savefig("../slides/model_comparison.png", dpi=150, bbox_inches="tight")
    print("Grafico salvo em slides/model_comparison.png")

def main():
    """
    Funcao principal que orquestra todo o pipeline de Machine Learning.

    Fluxo:
        1. Carregamento dos dados
        2. Exploracao inicial (shape, nulos, distribuicao)
        3. Geracao dos graficos de EDA
        4. Pre-processamento (separacao X/y, remocao de colunas inuteis)
        5. Configuracao do preprocessor (ColumnTransformer)
        6. Divisao treino/teste com stratified sampling
        7. Treinamento e avaliacao com GridSearchCV
        8. Exibicao dos resultados no terminal
        9. Geracao do grafico comparativo final
    """
    print("=" * 64)
    print("  PREVISAO DE ROTATIVIDADE DE FUNCIONARIOS")
    print("  Machine Learning - Classificacao Binaria (Turnover)")
    print("=" * 64)

    # Etapa 1: Carregamento
    df = load_data()

    # Etapa 2: Exploracao
    explore_data(df)

    # Etapa 3: Graficos EDA
    plot_eda(df)

    # Etapa 4: Separacao features / target
    X, y = preprocess_data(df)

    # Etapa 5: Configuracao do pre-processador
    # O ColumnTransformer e configurado ANTES da divisao treino/teste,
    # mas o metodo fit() sera chamado APENAS no treino dentro do Pipeline.
    preprocessor, num_feat, cat_feat = build_preprocessor(X)

    # Etapa 6: Divisao estratificada
    # O parametro stratify=y garante que a proporcao 84/16 das classes
    # seja mantida tanto no treino quanto no teste.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nDivisao: Treino={X_train.shape[0]}, Teste={X_test.shape[0]}")
    print(f"Attrition em treino:\n{y_train.value_counts(normalize=True).mul(100).round(2)}")

    # Etapa 7: Treinamento e avaliacao
    results = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)

    # Etapa 8: Exibicao dos resultados
    print("\n" + "=" * 64)
    print("RESULTADOS - CLASSIFICACAO BINARIA (TURNOVER)")
    print("=" * 64)
    for r in results:
        print(f"\n{'='*40}")
        print(f"Modelo: {r['model']}")
        print(f"Melhores hiperparametros: {r['best_params']}")
        print(f"Melhor F1 (CV): {r['best_score']:.4f}")
        print(f"Acuracia (Teste):  {r['accuracy']:.4f}")
        print(f"Precisao:  {r['precision']:.4f}")
        print(f"Recall:    {r['recall']:.4f}")
        print(f"F1-Score:  {r['f1_score']:.4f}")
        print(f"ROC-AUC:   {r['roc_auc']:.4f}")
        print(f"\nRelatorio:\n{r['classification_report']}")
        print(f"Matriz de Confusao:\n{r['confusion_matrix']}")

    # Etapa 9: Grafico comparativo final
    plot_results(results, y_test)
    print("\nProjeto concluido! Verifique os graficos em slides/")


if __name__ == "__main__":
    main()
