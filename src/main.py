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
import os
import json
import joblib
# Suprime warnings de convergencia e deprecacao para manter a saida limpa
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

# Caminho robusto absoluto baseado na localizacao deste arquivo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HR_Employee_Attrition.csv")
RANDOM_STATE = 42

COLUMNS_MAP = {
    'Age': 'Idade',
    'Attrition': 'Turnover',
    'BusinessTravel': 'ViagemNegocios',
    'DailyRate': 'ValorDiaria',
    'Department': 'Departamento',
    'DistanceFromHome': 'DistanciaTrabalho',
    'Education': 'Escolaridade',
    'EducationField': 'AreaFormacao',
    'EmployeeCount': 'ContagemFuncionarios',
    'EmployeeNumber': 'NumeroFuncionario',
    'EnvironmentSatisfaction': 'SatisfacaoAmbiente',
    'Gender': 'Genero',
    'HourlyRate': 'ValorHora',
    'JobInvolvement': 'EnvolvimentoTrabalho',
    'JobLevel': 'NivelCargo',
    'JobRole': 'Cargo',
    'JobSatisfaction': 'SatisfacaoCargo',
    'MaritalStatus': 'EstadoCivil',
    'MonthlyIncome': 'RendaMensal',
    'MonthlyRate': 'ValorMensal',
    'NumCompaniesWorked': 'NumeroEmpresasTrabalhou',
    'Over18': 'MaiorDe18',
    'OverTime': 'HoraExtra',
    'PercentSalaryHike': 'PercentualAumentoSalario',
    'PerformanceRating': 'AvaliacaoDesempenho',
    'RelationshipSatisfaction': 'SatisfacaoRelacionamento',
    'StandardHours': 'HorasPadrao',
    'StockOptionLevel': 'NivelOpcaoAcoes',
    'TotalWorkingYears': 'TotalAnosTrabalhados',
    'TrainingTimesLastYear': 'QtdTreinamentosAnoPassado',
    'WorkLifeBalance': 'EquilibrioVidaTrabalho',
    'YearsAtCompany': 'AnosEmpresa',
    'YearsInCurrentRole': 'AnosCargoAtual',
    'YearsSinceLastPromotion': 'AnosDesdeUltimaPromocao',
    'YearsWithCurrManager': 'AnosGerenteAtual'
}

def translate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Traduzir valores categoricos
    categorical_translation = {
        'Attrition': {'Yes': 'Sim', 'No': 'Não'},
        'OverTime': {'Yes': 'Sim', 'No': 'Não'},
        'BusinessTravel': {
            'Travel_Rarely': 'Rara',
            'Travel_Frequently': 'Frequente',
            'Non-Travel': 'Não Viaja'
        },
        'Gender': {'Male': 'Masculino', 'Female': 'Feminino'},
        'MaritalStatus': {
            'Single': 'Solteiro',
            'Married': 'Casado',
            'Divorced': 'Divorciado'
        },
        'Department': {
            'Sales': 'Vendas',
            'Research & Development': 'Pesquisa & Desenvolvimento',
            'Human Resources': 'Recursos Humanos'
        },
        'JobRole': {
            'Sales Executive': 'Executivo de Vendas',
            'Research Scientist': 'Cientista de Pesquisa',
            'Laboratory Technician': 'Técnico de Laboratório',
            'Manufacturing Director': 'Diretor de Manufatura',
            'Healthcare Representative': 'Representante de Saúde',
            'Manager': 'Gerente',
            'Sales Representative': 'Representante de Vendas',
            'Research Director': 'Diretor de Pesquisa',
            'Human Resources': 'Recursos Humanos'
        },
        'EducationField': {
            'Life Sciences': 'Ciências da Vida',
            'Medical': 'Medicina',
            'Marketing': 'Marketing',
            'Technical Degree': 'Curso Técnico',
            'Other': 'Outro',
            'Human Resources': 'Recursos Humanos'
        }
    }
    
    for col, mapping in categorical_translation.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
            
    df = df.rename(columns=COLUMNS_MAP)
    return df

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Carrega o dataset IBM HR a partir de um arquivo CSV e traduz para Portugues.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = translate_dataframe(df)
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
    attrition_dist = df["Turnover"].value_counts()
    attrition_pct = df["Turnover"].value_counts(normalize=True) * 100
    print(f"\nDistribuicao Turnover:\n{attrition_dist}\n{attrition_pct.round(2)}")
    return df

def plot_eda(df: pd.DataFrame) -> None:
    """
    Gera graficos de Analise Exploratoria de Dados (EDA) e salva como PNG.
    """
    # Cria uma figura 2x3 com tamanho 15x10 polegadas
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Analise Exploratoria de Dados (EDA)", fontsize=16)

    # (1,1) - Distribuicao da variavel alvo (balanceamento das classes)
    sns.countplot(data=df, x="Turnover", hue="Turnover", ax=axes[0, 0], legend=False)
    axes[0, 0].set_title("Distribuicao de Turnover")

    # (1,2) - Idade vs Turnover
    sns.boxplot(data=df, x="Turnover", y="Idade", hue="Turnover", ax=axes[0, 1], legend=False)
    axes[0, 1].set_title("Idade vs Turnover")

    # (1,3) - Renda mensal vs Turnover
    sns.boxplot(data=df, x="Turnover", y="RendaMensal", hue="Turnover", ax=axes[0, 2], legend=False)
    axes[0, 2].set_title("Renda Mensal vs Turnover")

    # (2,1) - Proporcao de turnover por departamento (barras empilhadas)
    att_by_dept = df.groupby("Departamento")["Turnover"].value_counts(normalize=True).unstack() * 100
    att_by_dept.plot(kind="bar", stacked=True, ax=axes[1, 0], color=["#4CAF50", "#F44336"])
    axes[1, 0].set_title("Turnover por Departamento (%)")
    axes[1, 0].set_ylabel("Percentual")
    axes[1, 0].legend(title="Turnover")

    # (2,2) - Horas extras vs Turnover
    sns.countplot(data=df, x="HoraExtra", hue="Turnover", ax=axes[1, 1])
    axes[1, 1].set_title("Hora Extra vs Turnover")

    # (2,3) - Turnover por Cargo
    sns.countplot(data=df, x="Cargo", hue="Turnover", ax=axes[1, 2])
    axes[1, 2].set_title("Turnover por Cargo")
    axes[1, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, "slides", "eda_plots.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Graficos EDA salvos em {output_path}")

def preprocess_data(df: pd.DataFrame):
    """
    Separa features (X) e target (y) e remove colunas irrelevantes.
    """
    X = df.drop(columns=["Turnover", "ContagemFuncionarios", "HorasPadrao", "MaiorDe18", "NumeroFuncionario"])
    y = df["Turnover"].map({"Sim": 1, "Não": 0})
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

    Modelo 3 - Gradient Boosting:
        - random_state=42: reprodutibilidade.

    Grids:
        - Reg. Logistica: busca o melhor C (regularizacao inversa) para
          controlar overfitting. Valores menores = mais regularizacao.
        - Random Forest: busca profundidade maxima, numero de arvores,
          e criterios de parada (min_samples_split, min_samples_leaf).
        - Gradient Boosting: busca n_estimators, learning_rate e max_depth.

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
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
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
        "GradientBoosting": {
            "classifier__n_estimators": [50, 100, 150],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__max_depth": [3, 5],
        },
    }
    return models, param_grids

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor):
    """
    Treina cada modelo com GridSearchCV e avalia nos conjuntos de validacao e teste.

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
        X_val: features de validacao.
        X_test: features de teste.
        y_train: rotulos de treino.
        y_val: rotulos de validacao.
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

        # Predicoes no conjunto de validacao
        y_val_pred = gs.predict(X_val)
        y_val_proba = gs.predict_proba(X_val)[:, 1]
        
        # Predicoes no conjunto de teste
        y_test_pred = gs.predict(X_test)
        y_test_proba = gs.predict_proba(X_test)[:, 1]

        # Obter feature importances se o classificador suportar
        classifier = gs.best_estimator_.named_steps["classifier"]
        feature_importances = None
        if hasattr(classifier, "feature_importances_"):
            feature_importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            feature_importances = classifier.coef_[0]

        # Obter feature names do preprocessor
        feature_names = gs.best_estimator_.named_steps["preprocessor"].get_feature_names_out()

        # Colecao de todas as metricas para comparacao posterior
        results.append({
            "model": name,
            "best_estimator": gs.best_estimator_,
            "best_params": gs.best_params_,
            "best_score": gs.best_score_,          # F1 medio da validacao cruzada
            
            # Validação
            "val_f1": f1_score(y_val, y_val_pred),
            "val_recall": recall_score(y_val, y_val_pred),
            "val_roc_auc": roc_auc_score(y_val, y_val_proba),

            # Teste (para plot_results)
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
    """
    Gera grafico comparativo entre os modelos e salva como PNG.

    Layout 2x3:
        Linha superior:
        - (0,0) - Barras comparativas de metricas (Acuracia, Precisao, Recall, F1, AUC)
        - (0,1) - Curva ROC sobreposta dos modelos
        - (0,2) - Top 10 features mais importantes (Random Forest)
        Linha inferior:
        - (1,0) - Matriz de confusao - LogisticRegression
        - (1,1) - Matriz de confusao - RandomForest
        - (1,2) - Matriz de confusao - GradientBoosting

    Args:
        results: lista de dicionarios com as metricas de cada modelo.
        y_test: rotulos verdadeiros do conjunto de teste.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Comparacao de Modelos - Previsao de Turnover", fontsize=16, fontweight="bold", y=0.98)

    # (0,0) - Grafico de barras com todas as metricas lado a lado
    metrics_df = pd.DataFrame([
        {"Model": r["model"], "Acuracia": r["accuracy"],
         "Precisao": r["precision"], "Recall": r["recall"],
         "F1-Score": r["f1_score"], "ROC-AUC": r["roc_auc"]}
        for r in results
    ]).set_index("Model")

    metrics_df.plot(kind="bar", ax=axes[0, 0], rot=0, colormap="viridis")
    axes[0, 0].set_title("Metricas de Desempenho", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylim(0, 1.05)      # Metricas sao proporcoes entre 0 e 1
    axes[0, 0].legend(loc="lower left", fontsize=8)
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].set_xlabel("")

    # (0,1) - Curva ROC
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        axes[0, 1].plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0, 1].set_title("Curva ROC", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Taxa de Falso Positivo")
    axes[0, 1].set_ylabel("Taxa de Verdadeiro Positivo")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # (0,2) - Importancia de features do Random Forest (top 10)
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
        fi_sorted = fi.argsort()[-10:][::-1]
        if feature_names is not None:
            cleaned_names = [str(feature_names[idx]).replace("cat__", "").replace("num__", "") for idx in fi_sorted]
            axes[0, 2].barh(range(10), fi[fi_sorted], color="steelblue")
            axes[0, 2].set_yticks(range(10))
            axes[0, 2].set_yticklabels(cleaned_names, fontsize=8)
        else:
            axes[0, 2].barh(range(10), fi[fi_sorted], color="steelblue")
            axes[0, 2].set_yticks(range(10))
            axes[0, 2].set_yticklabels(range(10))
        axes[0, 2].set_title("Top 10 Features - Random Forest", fontsize=12, fontweight="bold")
        axes[0, 2].invert_yaxis()
    else:
        axes[0, 2].text(0.5, 0.5, "Feature importance\nindisponivel",
                        ha="center", va="center")
        axes[0, 2].set_title("Importancia das Features", fontsize=12, fontweight="bold")

    # Linha inferior: Matrizes de confusao lado a lado
    for i, r in enumerate(results):
        ax = axes[1, i]
        sns.heatmap(r["confusion_matrix"], annot=True, fmt="d",
                    cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"Matriz de Confusao\n{r['model']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        ax.set_xticklabels(["Fica", "Sai"])
        ax.set_yticklabels(["Fica", "Sai"])

    # Ajusta o layout para evitar sobreposicoes
    plt.tight_layout()
    # Salva o grafico usando caminho robusto
    output_path = os.path.join(BASE_DIR, "slides", "model_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Grafico salvo em {output_path}")

def main():
    """
    Funcao principal que orquestra todo o pipeline de Machine Learning.

    Fluxo:
        1. Carregamento dos dados
        2. Exploracao inicial (shape, nulos, distribuicao)
        3. Geracao dos graficos de EDA
        4. Pre-processamento (separacao X/y, remocao de colunas inuteis)
        5. Configuracao do preprocessor (ColumnTransformer)
        6. Divisao treino/val/teste com stratified sampling (60/20/20)
        7. Treinamento e avaliacao com GridSearchCV
        8. Exibicao dos resultados no terminal e persistencia do melhor modelo
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
    preprocessor, num_feat, cat_feat = build_preprocessor(X)

    # Etapa 6: Divisao estratificada 60/20/20
    # 1. Separar 60% para treino e 40% intermediario
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    # 2. Dividir o intermediario em 50% val e 50% teste (resultando em 20% val e 20% teste do total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    
    print(f"\nDivisao (60/20/20): Treino={X_train.shape[0]}, Validacao={X_val.shape[0]}, Teste={X_test.shape[0]}")
    print(f"Turnover em treino:\n{y_train.value_counts(normalize=True).mul(100).round(2)}")

    # Etapa 7: Treinamento e avaliacao
    results = train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)

    # Etapa 8: Exibicao dos resultados e persistencia
    print("\n" + "=" * 64)
    print("RESULTADOS - CLASSIFICACAO BINARIA (TURNOVER)")
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

        # Selecao do melhor modelo baseado no F1 na Validacao
        if r['val_f1'] > best_val_f1:
            best_val_f1 = r['val_f1']
            best_model_name = r['model']
            best_pipeline = r['best_estimator']

    print(f"\n>>> Melhor modelo selecionado (Val F1={best_val_f1:.4f}): {best_model_name}")

    # Salva o melhor modelo final
    model_path = os.path.join(BASE_DIR, "modelo_turnover.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"Melhor modelo persistido em: {model_path}")

    # Salva métricas para preenchimento nos slides
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
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metricas exportadas para: {metrics_path}")

    # Etapa 9: Grafico comparativo final
    plot_results(results, y_test)
    print("\nProjeto concluido! Verifique os graficos em slides/")


if __name__ == "__main__":
    main()
