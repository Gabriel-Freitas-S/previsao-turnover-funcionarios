# Previsão de Rotatividade de Funcionários

Classificação Binária para Turnover Corporativo usando Machine Learning.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Concluído-green)]()

## Problema

Prever se um funcionário tem alta probabilidade de deixar a empresa (Turnover/Attrition).
Trata-se de um problema de **Classificação Binária** com dados desbalanceados (84% permanecem, 16% saem).

**Relevância:** A saída repentina de funcionários gera custos de recrutamento, perda de conhecimento
institucional e riscos à conformidade legal. A retenção preditiva de talentos é uma prioridade
estratégica para RH.

## Dataset

**IBM HR Analytics Employee Attrition & Performance** (Kaggle)

- 1470 registros, 35 atributos (26 numéricos, 9 categóricos)
- 0 valores nulos
- Variável alvo: `Attrition` (Yes/No)

## Metodologia

### Pré-processamento
- Remoção de colunas sem variância (`EmployeeCount`, `StandardHours`, `Over18`)
- Remoção do identificador (`EmployeeNumber`)
- Codificação One-Hot para 7 variáveis categóricas
- Padronização (StandardScaler) para variáveis numéricas
- ColumnTransformer + Pipeline para evitar data leakage
- Divisão: 80% treino / 20% teste (com stratified sampling)

### Modelos
| Modelo | Hiperparâmetros Otimizados |
|--------|---------------------------|
| **Regressão Logística** | C=0.1, penalty=L2, solver=lbfgs, class_weight='balanced' |
| **Random Forest** | n_estimators=100, max_depth=5, min_samples_leaf=2, class_weight='balanced' |

### Otimização
- GridSearchCV com validação cruzada StratifiedKFold (5 folds)
- Métrica de otimização: F1-Score (classe minoritária)
- Estratégia de balanceamento: `class_weight='balanced'`

## Resultados

| Métrica | Reg. Logística | Random Forest | Melhor |
|---------|:-------------:|:-------------:|:-----:|
| Acurácia | 0,7619 | 0,8231 | RF |
| Precisão | 0,3678 | 0,4468 | RF |
| Recall | **0,6809** | 0,4468 | RL |
| F1-Score | **0,4776** | 0,4468 | RL |
| ROC-AUC | **0,8022** | 0,7494 | RL |

A **Regressão Logística** foi o modelo mais adequado, priorizando a detecção de
funcionários em risco de saída (maior Recall e F1-Score).

## Como Reproduzir

### Pré-requisitos
- Python 3.11+
- Podman ou Docker

### Com Podman (recomendado)

```bash
# Clonar o repositório
git clone https://github.com/Gabriel-Freitas-S/previsao-turnover-funcionarios.git
cd previsao-turnover-funcionarios

# Executar o pipeline completo
podman run --rm --userns=keep-id \
  -v .:/workspaces/previsao-turnover-funcionarios \
  -w /workspaces/previsao-turnover-funcionarios/src \
  docker.io/jupyter/scipy-notebook \
  python3 main.py

# Gerar os slides
podman run --rm --userns=keep-id \
  -v .:/workspaces/previsao-turnover-funcionarios \
  -w /workspaces/previsao-turnover-funcionarios/src \
  docker.io/jupyter/scipy-notebook \
  bash -c "pip install -q -r ../requirements.txt && python3 generate_slides.py"
```

### Com Dev Container (Zed)

1. Abra a pasta do projeto no [Zed Editor](https://zed.dev)
2. Aceite "Reopen in Container" quando solicitado
3. No terminal do container:

```bash
python3 src/main.py
pip install -q -r requirements.txt && python3 src/generate_slides.py
```

### Jupyter Notebook

```bash
podman run --rm --userns=keep-id -p 8888:8888 \
  -v .:/workspaces/previsao-turnover-funcionarios \
  docker.io/jupyter/scipy-notebook
```

Abra o notebook em `notebooks/previsao_turnover.ipynb`.

## Estrutura do Projeto

```
previsao-turnover-funcionarios/
├── .devcontainer/
│   └── devcontainer.json      # Configuração do Dev Container (Podman)
├── data/
│   └── HR_Employee_Attrition.csv  # Dataset IBM HR
├── notebooks/
│   └── previsao_turnover.ipynb    # Jupyter Notebook interativo
├── slides/
│   ├── apresentacao_turnover.pdf  # Slides da apresentação (10 páginas)
│   ├── eda_plots.png              # Gráficos da análise exploratória
│   └── model_comparison.png       # Comparação dos modelos
├── src/
│   ├── main.py                 # Pipeline ML completo
│   └── generate_slides.py      # Geração de slides em PDF
├── SPEC/
│   └── Trabalho Machine Learning C3.md  # Especificações do trabalho
├── index.html                  # Página GitHub Pages
└── README.md                   # Este arquivo
```

## Tecnologias

- **Python 3.11** — scikit-learn, pandas, matplotlib, seaborn
- **Container** — jupyter/scipy-notebook via Podman
- **Editor** — Zed com Dev Containers
- **Slides** — fpdf2 (PDF)

## Licença

Projeto acadêmico — Machine Learning, Unidade 3.
