"""
================================================================================
PROJETO ACADÊMICO: PREVISÃO DE ROTATIVIDADE DE FUNCIONÁRIOS (TURNOVER)
================================================================================
Disciplina: Machine Learning / Aprendizado de Máquina
Abordagem: Classificação Binária Supervisionada (Ficou [0] vs. Saiu [1])

Este script implementa o ciclo de vida completo de um projeto de Machine Learning:
1. Carga de dados e análise exploratória de dados (EDA) para identificar padrões.
2. Engenharia de recursos e pré-processamento de dados estruturados.
3. Criação de pipelines robustas para evitar vazamento de dados (data leakage).
4. Treinamento de múltiplos classificadores de diferentes paradigmas:
   - Regressão Logística (modelo linear paramétrico de base)
   - Random Forest (conjunto homogêneo de árvores baseado em bagging)
   - Gradient Boosting (conjunto sequencial baseado em boosting)
5. Ajuste de hiperparâmetros por Busca em Grade (GridSearchCV) e Validação
   Cruzada Estratificada (Stratified K-Fold).
6. Avaliação rigorosa utilizando métricas apropriadas para dados desbalanceados,
   com foco especial no F1-Score (média harmônica entre Precisão e Revogação).
7. Geração automática de gráficos de resultados para fins acadêmicos e corporativos.
8. Persistência do melhor modelo para uso posterior (deploy em ambiente web/Pyodide).
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import joblib

# Desativar avisos desnecessários (como avisos de depreciação de bibliotecas)
warnings.filterwarnings("ignore")

# Importações específicas do scikit-learn para divisão de dados e validação
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# Pré-processadores: normalização de escala e conversão de variáveis textuais
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# ColumnTransformer permite aplicar transformações diferentes a colunas diferentes do DataFrame
from sklearn.compose import ColumnTransformer
# Pipeline ajuda a encadear passos de pré-processamento e estimação de forma sequencial
from sklearn.pipeline import Pipeline

# Modelos/Classificadores avaliados
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Métricas de avaliação de desempenho
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Definição de constantes globais do projeto para garantir reprodutibilidade
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HR_Analytics.csv")
RANDOM_STATE = 42  # Semente fixa para garantir que as divisões de dados e modelos sejam reprodutíveis


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Carrega o arquivo CSV que contém as informações dos funcionários.
    
    Usa 'utf-8-sig' para ler o arquivo corretamente, mesmo se tiver caracteres
    de marcação de ordem de byte (BOM) gerados por softwares como Microsoft Excel.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def explore_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza uma análise descritiva inicial do conjunto de dados no console.
    Exibe:
      - Dimensões do conjunto (linhas x colunas).
      - Nomes das variáveis disponíveis.
      - Contagem de tipos de dados (numéricos vs textuais).
      - Verificação de valores nulos/ausentes na base.
      - Distribuição e proporção da variável alvo 'saiu' para detectar desbalanceamento.
    """
    print(f"Dimensões do Dataset (Linhas x Colunas): {df.shape}")
    print(f"\nVariáveis (Colunas): {df.columns.tolist()}")
    print(f"\nContagem de Tipos de Dados:\n{df.dtypes.value_counts()}")
    
    # É fundamental checar nulos para saber se precisaremos de técnicas de imputação (ex: média ou mediana)
    total_nulos = df.isnull().sum().sum()
    print(f"\nTotal de Valores Nulos no Dataset: {total_nulos}")
    
    # Análise de desbalanceamento de classe: se uma classe for muito majoritária,
    # a acurácia simples pode ser uma métrica enganosa (paradoxo da acurácia).
    target_dist = df["saiu"].value_counts()
    target_pct = df["saiu"].value_counts(normalize=True) * 100
    print(f"\nDistribuição da Variável Alvo 'saiu' (0 = Ficou, 1 = Saiu):")
    print(f"Frequência Absoluta:\n{target_dist}")
    print(f"Frequência Relativa (%):\n{target_pct.round(2)}")
    return df


def plot_eda(df: pd.DataFrame) -> None:
    """
    Gera e salva gráficos detalhados de Análise Exploratória de Dados (EDA).
    
    Os gráficos ajudam a criar hipóteses visuais sobre as variáveis explicativas 
    que mais afetam a decisão de saída dos colaboradores (Turnover).
    Salva uma imagem combinada e imagens separadas para serem usadas nos slides do projeto.
    """
    # Criamos uma figura com uma grade de 2 linhas e 3 colunas para consolidar a análise
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Análise Exploratória de Dados (EDA) — HR Analytics", fontsize=16)

    # Copiamos o DataFrame para formatar os rótulos de forma legível no gráfico
    df_plot = df.copy()
    df_plot["Rotatividade"] = df_plot["saiu"].map({0: "Não", 1: "Sim"})

    # Grafico 1: Distribuição da Variável Alvo (Turnover)
    # Mostra visualmente a proporção de funcionários que saíram vs. os que ficaram.
    sns.countplot(data=df_plot, x="Rotatividade", hue="Rotatividade",
                  ax=axes[0, 0], legend=False, palette=["#4CAF50", "#F44336"])
    axes[0, 0].set_title("Distribuição Geral de Turnover")
    axes[0, 0].set_xlabel("")

    # Grafico 2: Boxplot do Nível de Satisfação vs Turnover
    # Permite analisar se funcionários que saíram apresentavam menor satisfação mediana.
    sns.boxplot(data=df_plot, x="Rotatividade", y="nivel_satisfacao", hue="Rotatividade",
                ax=axes[0, 1], legend=False, palette=["#4CAF50", "#F44336"])
    axes[0, 1].set_title("Nível de Satisfação vs Turnover")
    axes[0, 1].set_ylabel("Nível de Satisfação [0 a 1]")

    # Grafico 3: Boxplot da Média de Horas Mensais vs Turnover
    # Mostra se há relação de sobrecarga de trabalho (muitas horas) com a evasão.
    sns.boxplot(data=df_plot, x="Rotatividade", y="media_horas_mensais", hue="Rotatividade",
                ax=axes[0, 2], legend=False, palette=["#4CAF50", "#F44336"])
    axes[0, 2].set_title("Horas Mensais Trabalhadas vs Turnover")
    axes[0, 2].set_ylabel("Média de Horas Mensais")

    # Grafico 4: Percentual de Turnover por Departamento (Barras Empilhadas 100%)
    # Útil para identificar se setores específicos (ex: vendas, suporte) têm maiores taxas de saída.
    att_by_dept = df_plot.groupby("departamento")["saiu"].value_counts(normalize=True).unstack() * 100
    if 1 in att_by_dept.columns:
        att_by_dept = att_by_dept[[1, 0]] if 0 in att_by_dept.columns else att_by_dept
    else:
        att_by_dept[1] = 0
        att_by_dept[0] = 100 - att_by_dept[1]
    att_by_dept = att_by_dept.sort_values(1, ascending=False)
    att_by_dept.plot(kind="bar", stacked=True, ax=axes[1, 0], color=["#F44336", "#4CAF50"])
    axes[1, 0].set_title("Rotatividade por Departamento (%)")
    axes[1, 0].set_ylabel("Percentual")
    axes[1, 0].legend(["Saiu", "Ficou"], title="Rotatividade")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Grafico 5: Distribuição do Turnover por Faixa Salarial
    # Demonstra o impacto do fator econômico (salário baixo, médio ou alto) na evasão.
    sal_order = ["baixo", "medio", "alto"]
    sns.countplot(data=df_plot, x="salario", hue="Rotatividade",
                  ax=axes[1, 1], order=sal_order, palette=["#4CAF50", "#F44336"])
    axes[1, 1].set_title("Faixa Salarial vs Turnover")
    axes[1, 1].set_xlabel("Nível Salarial")

    # Grafico 6: Boxplot da Última Avaliação de Desempenho vs Turnover
    # Examina se quem sai tem avaliação baixa (performance ruim) ou muito alta (talentos buscando novos ares).
    sns.boxplot(data=df_plot, x="Rotatividade", y="ultima_avaliacao", hue="Rotatividade",
                ax=axes[1, 2], legend=False, palette=["#4CAF50", "#F44336"])
    axes[1, 2].set_title("Última Avaliação de Desempenho vs Turnover")
    axes[1, 2].set_ylabel("Pontuação da Avaliação [0 a 1]")

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, "slides", "eda_plots.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico consolidado de EDA salvo em: {output_path}")

    # -------------------------------------------------------------------------
    # Geração dos gráficos individuais para montagem de slides da apresentação
    # -------------------------------------------------------------------------
    slides_dir = os.path.join(BASE_DIR, "slides")
    
    # 1. Distribuição de Turnover
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.countplot(data=df_plot, x="Rotatividade", hue="Rotatividade",
                  ax=ax, legend=False, palette=["#4CAF50", "#F44336"])
    ax.set_title("Distribuição Geral de Turnover")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(slides_dir, "eda_turnover_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Satisfação
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.boxplot(data=df_plot, x="Rotatividade", y="nivel_satisfacao", hue="Rotatividade",
                ax=ax, legend=False, palette=["#4CAF50", "#F44336"])
    ax.set_title("Satisfação vs Turnover")
    ax.set_ylabel("Nível de Satisfação")
    plt.tight_layout()
    plt.savefig(os.path.join(slides_dir, "eda_satisfacao.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Horas Mensais
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.boxplot(data=df_plot, x="Rotatividade", y="media_horas_mensais", hue="Rotatividade",
                ax=ax, legend=False, palette=["#4CAF50", "#F44336"])
    ax.set_title("Horas Trabalhadas vs Turnover")
    ax.set_ylabel("Média de Horas Mensais")
    plt.tight_layout()
    plt.savefig(os.path.join(slides_dir, "eda_horas.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Departamento
    fig, ax = plt.subplots(figsize=(5, 3.5))
    att_by_dept.plot(kind="bar", stacked=True, ax=ax, color=["#F44336", "#4CAF50"])
    ax.set_title("Turnover por Departamento (%)")
    ax.set_ylabel("Percentual")
    ax.legend(["Saiu", "Ficou"], title="Rotatividade")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(slides_dir, "eda_dept.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Salário
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.countplot(data=df_plot, x="salario", hue="Rotatividade",
                  ax=ax, order=sal_order, palette=["#4CAF50", "#F44336"])
    ax.set_title("Faixa Salarial vs Turnover")
    ax.set_xlabel("Nível Salarial")
    plt.tight_layout()
    plt.savefig(os.path.join(slides_dir, "eda_salario.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 6. Avaliação
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.boxplot(data=df_plot, x="Rotatividade", y="ultima_avaliacao", hue="Rotatividade",
                ax=ax, legend=False, palette=["#4CAF50", "#F44336"])
    ax.set_title("Avaliação de Desempenho vs Turnover")
    ax.set_ylabel("Pontuação da Avaliação")
    plt.tight_layout()
    plt.savefig(os.path.join(slides_dir, "eda_avaliacao.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("Gráficos de EDA individuais salvos com sucesso em slides/")


def preprocess_data(df: pd.DataFrame):
    """
    Separa o conjunto de dados em variáveis preditoras (X) e variável alvo (y).
    
    Também limpa possíveis espaços em branco nas extremidades de variáveis de texto.
    """
    df_proc = df.copy()

    # Variável alvo (target): convertida explicitamente para inteiro (0 ou 1)
    y = df_proc["saiu"].astype(int)

    # Variáveis preditoras (features): removemos apenas a coluna alvo para evitar vazamento
    cols_to_drop = ["saiu"]
    X = df_proc.drop(columns=[c for c in cols_to_drop if c in df_proc.columns], errors="ignore")

    # Limpeza de strings: remove espaços em branco extras que podem comprometer o One-Hot Encoding
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].str.strip()

    return X, y


def build_preprocessor(X: pd.DataFrame):
    """
    Estrutura o pré-processamento de dados de forma modular usando ColumnTransformer.
    
    A separação é feita baseando-se no tipo de dado:
      - Variáveis Numéricas Contínuas:
        Passam por normalização escalar (StandardScaler), que centraliza a média
        em 0 e desvio padrão em 1. Essencial para que modelos lineares (como a 
        Regressão Logística) não priorizem atributos com magnitudes maiores 
        (ex: horas mensais vs. nível de satisfação).
      - Variáveis Categóricas Nominais:
        Convertidas via OneHotEncoder em variáveis binárias (dummy).
        O argumento 'drop="first"' é usado para remover a primeira coluna dummy
        gerada de cada categoria, eliminando a multicolinearidade estrutural
        (armadilha da variável dummy), crítica em modelos estatísticos e lineares.
    """
    # Identifica dinamicamente as colunas numéricas e categóricas
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    
    print(f"\nVariáveis Numéricas detectadas ({len(numeric_features)}): {numeric_features}")
    print(f"Variáveis Categóricas detectadas ({len(categorical_features)}): {categorical_features}")

    # Pipeline de processamento numérico
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    
    # Pipeline de processamento categórico
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    # Consolidação dos transformadores em um único processador mapeado por colunas
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return preprocessor, numeric_features, categorical_features


def create_models():
    """
    Define os classificadores avaliados no projeto e suas grades de busca de parâmetros.
    
    Explicação Didática dos Modelos Escolhidos:
      1. Regressão Logística (LogisticRegression):
         Modelo linear simples e altamente interpretável que calcula o logaritmo 
         das chances de turnover usando a função logística (sigmoide). Serve 
         como excelente baseline de comparação. Usa class_weight='balanced'.
      2. Floresta Aleatória (RandomForestClassifier):
         Modelo ensemble baseado em Bagging (Bootstrap Aggregating) que cria
         múltiplas árvores de decisão independentes treinadas em subamostras dos
         dados e faz a votação majoritária para a predição. Não linear e muito
         robusto a outliers. Usa class_weight='balanced'.
      3. Aumento de Gradiente (GradientBoostingClassifier):
         Modelo ensemble baseado em Boosting, onde árvores de decisão rasas são
         treinadas sequencialmente, com cada nova árvore focada em corrigir os
         erros residuais deixados pelo conjunto de árvores anterior. Geralmente 
         apresenta alta performance preditiva.
         
    Ajuste de Pesos ('class_weight="balanced"'):
      Utilizado na Regressão Logística e Random Forest para penalizar de forma
      mais severa os erros cometidos na classe minoritária (quem saiu da empresa),
      corrigindo o viés gerado pelo desbalanceamento do conjunto de dados.
    """
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
    
    # Hiperparâmetros otimizados via busca em grade (GridSearchCV)
    param_grids = {
        "RegressaoLogistica": {
            "classifier__C": [0.01, 0.1, 1, 10],  # Inverso da força de regularização L2 (menores valores = maior regularização)
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
        },
        "RandomForest": {
            "classifier__n_estimators": [100, 200],  # Número de árvores na floresta
            "classifier__max_depth": [5, 10, None],   # Profundidade máxima de cada árvore
            "classifier__min_samples_split": [2, 5],  # Mínimo de amostras para dividir um nó interno
            "classifier__min_samples_leaf": [1, 2],   # Mínimo de amostras exigidas em uma folha (nó terminal)
        },
        "GradientBoosting": {
            "classifier__n_estimators": [50, 100],     # Quantidade de etapas de boosting a executar
            "classifier__learning_rate": [0.05, 0.1, 0.2],  # Taxa de aprendizado que encolhe a contribuição de cada árvore
            "classifier__max_depth": [3, 5],            # Profundidade máxima de cada árvore estimadora
        },
    }
    return models, param_grids


def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor):
    """
    Treina os modelos usando Validação Cruzada Estratificada e avalia seus desempenhos.
    
    Didática da Validação Cruzada Estratificada (Stratified K-Fold):
      Dividimos os dados de treino em K partes (n_splits=5). O termo 'Estratificada'
      significa que cada partição (fold) mantém a exata mesma proporção original da
      variável alvo (saiu: ~24% sim / ~76% não). Isso evita folds contendo apenas
      uma classe, garantindo estabilidade e medição real do erro de generalização.
      
    Uso de Pipelines dentro da busca:
      O pré-processamento é inserido na pipeline ("preprocessor" + "classifier").
      Isso garante que transformações como cálculo de média e desvio padrão sejam
      feitas apenas nos dados de treino de cada iteração da validação cruzada,
      evitando o vazamento de dados de validação para o treinamento.
    """
    models, param_grids = create_models()
    results = []

    for name, model in models.items():
        print(f"\n>>> Ajustando hiperparâmetros para {name} via GridSearchCV...")

        # Montagem da pipeline: pré-processamento seguido do estimador
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        # Criação do objeto de busca em grade
        gs = GridSearchCV(
            pipe, param_grids[name],
            # K-Fold Estratificado com 5 folds garante validação confiável
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            # Otimizamos pelo F1-Score pois queremos precisão e recall equilibrados
            scoring="f1",
            n_jobs=-1,  # Utiliza todas as CPUs disponíveis em paralelo para acelerar o treino
            verbose=1
        )
        # Ajusta o modelo encontrando a melhor combinação de parâmetros
        gs.fit(X_train, y_train)

        # Predições na partição de Validação (usada para selecionar o melhor modelo final)
        y_val_pred = gs.predict(X_val)
        y_val_proba = gs.predict_proba(X_val)[:, 1]

        # Predições na partição de Teste (simulação de produção, dados nunca vistos)
        y_test_pred = gs.predict(X_test)
        y_test_proba = gs.predict_proba(X_test)[:, 1]

        # Extração de coeficientes (para regressão logística) ou importância de variáveis (para árvores)
        classifier = gs.best_estimator_.named_steps["classifier"]
        feature_importances = None
        if hasattr(classifier, "feature_importances_"):
            feature_importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            feature_importances = classifier.coef_[0]

        # Obtém os nomes das variáveis após o processamento do One-Hot Encoder
        feature_names = gs.best_estimator_.named_steps["preprocessor"].get_feature_names_out()

        # Armazena os resultados detalhados de treino, validação e teste
        results.append({
            "model": name,
            "best_estimator": gs.best_estimator_,
            "best_params": gs.best_params_,
            "best_score": gs.best_score_, # F1-score médio obtido na validação cruzada de treino

            # Métricas no conjunto de Validação
            "val_f1": f1_score(y_val, y_val_pred),
            "val_recall": recall_score(y_val, y_val_pred),
            "val_roc_auc": roc_auc_score(y_val, y_val_proba),

            # Métricas no conjunto de Teste Final
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
    Gera e salva visualizações consolidadas de comparação de performance dos modelos.
    
    Gráficos gerados:
      - Gráfico de Barras: Comparativo das principais métricas de teste de todos os modelos.
      - Curva ROC (Receiver Operating Characteristic): Gráfico que mede a capacidade de
        discriminação do modelo em diferentes limiares de decisão (eixo X: falsos positivos,
        eixo Y: verdadeiros positivos). Quanto maior a Área Sob a Curva (AUC), melhor o modelo.
      - Importância dos Atributos (Feature Importance): Mostra quais variáveis foram mais
        decisivas para as tomadas de decisão da Random Forest.
      - Matrizes de Confusão: Gráficos de calor mostrando a relação exata entre previsões
        e valores reais (verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos).
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Comparação de Modelos — Previsão de Turnover (HR Analytics)", fontsize=16, fontweight="bold", y=0.98)

    # 1. Gráfico de Barras Comparativo das Métricas
    metrics_df = pd.DataFrame([
        {"Modelo": r["model"], "Acurácia": r["accuracy"],
         "Precisão": r["precision"], "Recall": r["recall"],
         "F1-Score": r["f1_score"], "ROC-AUC": r["roc_auc"]}
        for r in results
    ]).set_index("Modelo")

    metrics_df.plot(kind="bar", ax=axes[0, 0], rot=0, colormap="viridis")
    axes[0, 0].set_title("Métricas de Desempenho no Teste", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].set_xlabel("")

    # 2. Curva ROC Comparativa
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        axes[0, 1].plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.3)  # Reta de referência aleatória
    axes[0, 1].set_title("Curva ROC Comparativa", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Taxa de Falso Positivo (FPR)")
    axes[0, 1].set_ylabel("Taxa de Verdadeiro Positivo (TPR)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # 3. Importância de Atributos (Random Forest)
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
            # Limpa prefixos gerados pelos pipelines para tornar os rótulos amigáveis
            cleaned_names = [str(feature_names[idx]).replace("cat__", "").replace("num__", "") for idx in fi_sorted]
            axes[0, 2].barh(range(n_top), fi[fi_sorted], color="steelblue")
            axes[0, 2].set_yticks(range(n_top))
            axes[0, 2].set_yticklabels(cleaned_names, fontsize=8)
        else:
            axes[0, 2].barh(range(n_top), fi[fi_sorted], color="steelblue")
            axes[0, 2].set_yticks(range(n_top))
            axes[0, 2].set_yticklabels(range(n_top))
        axes[0, 2].set_title(f"Top {n_top} Atributos Decisivos (Random Forest)", fontsize=12, fontweight="bold")
        axes[0, 2].invert_yaxis()
    else:
        axes[0, 2].text(0.5, 0.5, "Importância de atributos\nindisponível", ha="center", va="center")
        axes[0, 2].set_title("Importância dos Atributos", fontsize=12, fontweight="bold")

    # 4. Matrizes de Confusão Individuais
    for i, r in enumerate(results):
        ax = axes[1, i]
        sns.heatmap(r["confusion_matrix"], annot=True, fmt="d",
                    cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"Matriz de Confusão\n{r['model']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Classificação Prevista")
        ax.set_ylabel("Classe Real")
        ax.set_xticklabels(["Ficou (0)", "Saiu (1)"])
        ax.set_yticklabels(["Ficou (0)", "Saiu (1)"])

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, "slides", "model_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico consolidado de comparação salvo em: {output_path}")

    # -------------------------------------------------------------------------
    # Salva versões separadas dos gráficos para exibição nos slides acadêmicos
    # -------------------------------------------------------------------------
    slides_dir = os.path.join(BASE_DIR, "slides")
    model_names = {"RegressaoLogistica": "Regressão Logística",
                   "RandomForest": "Random Forest", "GradientBoosting": "Gradient Boosting"}

    # 1. Comparativo de Métricas
    fig, ax = plt.subplots(figsize=(5.5, 4))
    metrics_df.plot(kind="bar", ax=ax, rot=0, colormap="viridis")
    ax.set_title("Comparativo de Métricas de Desempenho", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(slides_dir, "comp_metricas.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Comparativo de Curvas ROC
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, label=f"{model_names.get(r['model'], r['model'])} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title("Comparação de Curvas ROC", fontsize=12, fontweight="bold")
    ax.set_xlabel("Taxa de Falso Positivo (FPR)")
    ax.set_ylabel("Taxa de Verdadeiro Positivo (TPR)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(slides_dir, "comp_roc.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Importância de Variáveis
    if fi_data:
        _, fi = fi_data[0]
        n_top = min(10, len(fi))
        fi_sorted = fi.argsort()[-n_top:][::-1]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        if feature_names is not None:
            cleaned_names = [str(feature_names[idx]).replace("cat__", "").replace("num__", "") for idx in fi_sorted]
            ax.barh(range(n_top), fi[fi_sorted], color="steelblue")
            ax.set_yticks(range(n_top))
            ax.set_yticklabels(cleaned_names, fontsize=8)
        else:
            ax.barh(range(n_top), fi[fi_sorted], color="steelblue")
            ax.set_yticks(range(n_top))
            ax.set_yticklabels(range(n_top))
        ax.set_title(f"Importância de Atributos — Random Forest", fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(slides_dir, "comp_features.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # 4. Matrizes de Confusão Individuais
    cm_names = [("RegressaoLogistica", "cm_logistica.png"),
                ("RandomForest", "cm_rf.png"),
                ("GradientBoosting", "cm_gb.png")]
    for model_key, fname in cm_names:
        for r in results:
            if r["model"] == model_key:
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(r["confusion_matrix"], annot=True, fmt="d",
                            cmap="Blues", ax=ax, cbar=False)
                ax.set_title(f"Matriz de Confusão\n{model_names.get(model_key, model_key)}",
                             fontsize=11, fontweight="bold")
                ax.set_xlabel("Classificação Prevista")
                ax.set_ylabel("Classe Real")
                ax.set_xticklabels(["Ficou", "Saiu"])
                ax.set_yticklabels(["Ficou", "Saiu"])
                plt.tight_layout()
                plt.savefig(os.path.join(slides_dir, fname), dpi=150, bbox_inches="tight")
                plt.close()
                break

    print("Gráficos comparativos individuais salvos com sucesso em slides/")


def main():
    """
    Fluxo principal que orquestra todo o projeto de modelagem preditiva de turnover.
    """
    print("=" * 72)
    print("        INICIANDO PIPELINE DE MACHINE LEARNING — CLASSIFICAÇÃO")
    print("   Tema: Previsão de Rotatividade de Funcionários (HR Analytics)")
    print("=" * 72)

    # 1. Carregar dados originais
    df = load_data()

    # 2. Estatística Descritiva Inicial
    explore_data(df)

    # 3. Análise Exploratória Gráfica
    plot_eda(df)

    # 4. Limpeza e Divisão entre Features (X) e Target (y)
    X, y = preprocess_data(df)

    # 5. Estruturação do Pré-processador (Escalonamento + Codificação)
    preprocessor, num_feat, cat_feat = build_preprocessor(X)

    # 6. Particionamento Didático da Base (Treino, Validação e Teste)
    #
    # Por que dividir em 3 partições (60/20/20)?
    #   - Base de Treino (60%): Usada para treinar os algoritmos e otimizar parâmetros via CV.
    #   - Base de Validação (20%): Usada para selecionar o melhor algoritmo comparando seus resultados.
    #   - Base de Teste (20%): Usada como avaliação final 'cega'. Garante que mediremos o erro de
    #     generalização sem viés de seleção de modelo.
    #
    # Usar 'stratify=y' é crucial para bases desbalanceadas para que as partições tenham as mesmas
    # proporções de funcionários que saíram (~23,6%) do conjunto original.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"\nDivisão Estatística dos Dados:")
    print(f"  - Conjunto de Treino:   {X_train.shape[0]} registros")
    print(f"  - Conjunto de Validação: {X_val.shape[0]} registros")
    print(f"  - Conjunto de Teste:     {X_test.shape[0]} registros")
    print(f"Proporção de Turnover em Treino:\n{y_train.value_counts(normalize=True).mul(100).round(2)}")

    # 7. Treinamento e Avaliação de Todos os Modelos com GridSearchCV
    results = train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)

    # 8. Exibição e Seleção do Melhor Modelo
    print("\n" + "=" * 72)
    print("                 RELATÓRIO COMPARATIVO DE PERFORMANCE")
    print("=" * 72)

    best_model_name = None
    best_val_f1 = -1.0
    best_pipeline = None

    for r in results:
        print(f"\n{'='*40}")
        print(f"Modelo: {r['model']}")
        print(f"  Hiperparâmetros Selecionados: {r['best_params']}")
        print(f"  F1-Score Médio (CV Treino):     {r['best_score']:.4f}")
        print(f"--- Métricas no Conjunto de Validação ---")
        print(f"  F1-Score (Validação):           {r['val_f1']:.4f}")
        print(f"  Recall/Revogação (Validação):   {r['val_recall']:.4f}")
        print(f"  Área sob Curva ROC (Validação): {r['val_roc_auc']:.4f}")
        print(f"--- Avaliação Cega (Conjunto de Teste) ---")
        print(f"  Acurácia (Teste):               {r['accuracy']:.4f}")
        print(f"  Precisão (Teste):               {r['precision']:.4f}")
        print(f"  Recall/Revogação (Teste):       {r['recall']:.4f}")
        print(f"  F1-Score (Teste):               {r['f1_score']:.4f}")
        print(f"  Área sob Curva ROC (Teste):     {r['roc_auc']:.4f}")
        print(f"\nRelatório de Classificação Completo (Teste):\n{r['classification_report']}")
        print(f"Matriz de Confusão (Teste):\n{r['confusion_matrix']}")

        # Selecionamos o melhor modelo baseado na performance do conjunto de VALIDAÇÃO.
        # Isso garante que não escolheremos baseado no teste, o que causaria vazamento de teste.
        if r['val_f1'] > best_val_f1:
            best_val_f1 = r['val_f1']
            best_model_name = r['model']
            best_pipeline = r['best_estimator']

    print(f"\n>>> VENCEDOR: {best_model_name} (Selecionado com F1 de Validação = {best_val_f1:.4f})")

    # 9. Persistência do Modelo Vencedor (Serialização com joblib)
    #
    # Salvamos o pipeline inteiro ajustado (que inclui o pré-processador ColumnTransformer
    # e o classificador vencedor). Isso garante robustez de ponta a ponta.
    model_path = os.path.join(BASE_DIR, "modelo_turnover.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"Pipeline do melhor modelo persistido com sucesso em: {model_path}")

    # 10. Exportar as métricas para JSON
    #
    # Salvamos os dados em formato estruturado para serem lidos dinamicamente pela interface web.
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
    print(f"Métricas reais do modelo exportadas para: {metrics_path}")

    # 11. Plotagem Gráfica
    plot_results(results, y_test)
    print("\nExecução completa do Pipeline Acadêmico de ML concluída! Verifique os gráficos em slides/")


if __name__ == "__main__":
    main()
