"""
================================================================================
SCRIPT ACADÊMICO: EXPORTAÇÃO DO MODELO DE TURNOVER PARA PYODIDE (WEB RUNTIME)
================================================================================
Problema Técnico:
    Modelos salvos com scikit-learn (.pkl) dependem da versão exata do scikit-learn
    e de outras bibliotecas do Python (como joblib) no ambiente de execução.
    Para executar o modelo no navegador do usuário (Client-side) usando Pyodide,
    podemos enfrentar incompatibilidades de versão ou lentidão ao instalar
    pacotes complexos no navegador.

Solução Didática:
    Este script extrai apenas os parâmetros matemáticos puros (pesos, médias,
    limiares de árvores de decisão) do pipeline treinado no scikit-learn e os
    salva como estruturas nativas do Python (dicionários e listas).
    
    Essas estruturas são salvas em um arquivo pickle leve (modelo_turnover_pyodide.pkl).
    Com isso, a engine de inferência na página web (index.html) pode carregar os pesos
    rapidamente e reconstruir a predição executando o cálculo manual (Numpy nativo),
    sem precisar do scikit-learn completo!

Extrações efetuadas:
    1. StandardScaler: médias e desvios padrões das colunas numéricas.
    2. OneHotEncoder: categorias detectadas e colunas descartadas (drop="first").
    3. GradientBoostingClassifier: coeficientes, caminhos de decisão (nós esquerdos,
       nós direitos, features de corte e limiares das árvores) e o prior inicial.
================================================================================
"""

import joblib
import pickle
import numpy as np
from pathlib import Path

# Definição dos caminhos das pastas do projeto
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "modelo_turnover.pkl"
DST = ROOT / "modelo_turnover_pyodide.pkl"

print("Iniciando conversão do modelo persistido para formato Pyodide nativo...")

# 1. Carrega o pipeline completo treinado pelo main.py
pipeline = joblib.load(SRC)

# 2. Desmembra o pipeline scikit-learn
ct = pipeline.named_steps["preprocessor"]      # Etapa de pré-processamento
gbc = pipeline.named_steps["classifier"]       # Classificador treinado (Gradient Boosting)

# 3. Separa os processadores das variáveis numéricas e categóricas
num_pipeline = ct.named_transformers_["num"]
cat_pipeline = ct.named_transformers_["cat"]
scaler = num_pipeline.named_steps["scaler"]
encoder = cat_pipeline.named_steps["onehot"]

# 4. Extração do Normalizador (StandardScaler)
# Extraímos a média e o desvio padrão de cada atributo numérico para refazer o escalonamento manual:
# Z = (X - média) / desvio_padrão
scaler_data = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
}
print("- Parâmetros do StandardScaler extraídos com sucesso.")

# 5. Extração do Codificador Categórico (OneHotEncoder)
# Extraímos as listas de categorias e qual índice foi removido (para reconstruir as colunas dummy manualmente)
encoder_data = {
    "categories": [c.tolist() for c in encoder.categories_],
    "drop_idx": encoder.drop_idx_[0],
}
print("- Parâmetros do OneHotEncoder extraídos com sucesso.")

# 6. Extração das Árvores de Decisão do Gradient Boosting
# O Gradient Boosting é composto por um conjunto sequencial de árvores de regressão simples.
# Extraímos as regras lógicas de cada nó de cada árvore para rodar a inferência no navegador.
trees = []
for est in gbc.estimators_:
    t = est[0].tree_
    trees.append({
        # Vetor contendo o índice do filho esquerdo de cada nó (-1 se for folha)
        "children_left": t.children_left.tolist(),
        # Vetor contendo o índice do filho direito de cada nó (-1 se for folha)
        "children_right": t.children_right.tolist(),
        # Índice do atributo usado para fazer a divisão (split) em cada nó
        "feature": t.feature.tolist(),
        # Limiar (valor de corte) para a divisão em cada nó
        "threshold": t.threshold.tolist(),
        # Valor de predição folha final associado a cada nó (formato: [node_count, 1, 1])
        "value": t.value.tolist(),
    })

# Cálculo do valor bruto da probabilidade inicial (prior log-odds da classe 1)
# O Gradient Boosting inicia sua inferência com uma probabilidade constante de base.
prior = gbc.init_.class_prior_
init_raw = float(np.log(prior[1] / prior[0]))

# Consolidação dos parâmetros do Gradient Boosting
gbc_data = {
    "init_raw": init_raw,
    "learning_rate": gbc.learning_rate,  # Fator de encolhimento de cada árvore
    "trees": trees,
}
print(f"- Parâmetros de {len(trees)} árvores do Gradient Boosting extraídos com sucesso.")

# 7. Consolidação Final do Modelo Portável
model_data = {
    "scaler": scaler_data,
    "encoder": encoder_data,
    "gbc": gbc_data,
}

# 8. Salvamento em arquivo binário compactado usando protocolo 4 (compatível com Pyodide/Python 3.8+)
with open(DST, "wb") as f:
    pickle.dump(model_data, f, protocol=4)

print(f"Conversão concluída! Novo modelo salvo em: {DST} ({DST.stat().st_size / 1024:.1f} KB)")
