# Previsão de Rotatividade de Funcionários

Classificação Binária para Turnover Corporativo usando Machine Learning.

**Integrantes do Grupo:**
- Gabriel Freitas Souza
- Indyanny Rodrigues Peixinho

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Concluído-green)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gabriel-Freitas-S/previsao-turnover-funcionarios/blob/main/notebooks/previsao_turnover.ipynb)

## Problema

Prever se um funcionário tem alta probabilidade de deixar a empresa (Turnover/Attrition).
Trata-se de um problema de **Classificação Binária** com dados desbalanceados (84% permanecem, 16% saem).

**Relevância:** A saída repentina de funcionários gera custos de recrutamento, perda de conhecimento
institucional e riscos à conformidade legal. A retenção preditiva de talentos é uma prioridade
estratégica para RH.

## Dataset

**IBM HR Analytics Employee Attrition & Performance** (Kaggle)

- 1470 registros, 35 atributos (23 numéricos, 7 categóricos restantes no X)
- 0 valores nulos
- Variável alvo: `Turnover` (Sim/Não)

## Metodologia

### Pré-processamento
- Remoção de colunas sem variância (`ContagemFuncionarios`, `HorasPadrao`, `MaiorDe18`)
- Remoção do identificador (`NumeroFuncionario`)
- Codificação One-Hot para 7 variáveis categóricas
- Padronização (StandardScaler) para variáveis numéricas
- ColumnTransformer + Pipeline para evitar data leakage
- Divisão: **60% treino / 20% validação / 20% teste** (estratificada)

### Modelos
| Modelo | Hiperparâmetros Otimizados |
|--------|---------------------------|
| **Regressão Logística** | C=0.1, penalty=L2, solver=lbfgs, class_weight='balanced' |
| **Random Forest** | n_estimators=200, max_depth=5, min_samples_leaf=1, min_samples_split=5, class_weight='balanced' |
| **Gradient Boosting** | learning_rate=0.1, max_depth=3, n_estimators=150 | (Nota: n_estimators otimizado para 100 com learning_rate 0.2 na execução PT) |

### Otimização
- GridSearchCV com validação cruzada StratifiedKFold (5 folds) no conjunto de treino
- Seleção do melhor modelo baseada no F1-Score do conjunto de validação
- Métrica de otimização: F1-Score (classe minoritária)
- Estratégia de balanceamento: `class_weight='balanced'` (onde aplicável)

## Resultados (Avaliação no Teste Final)

| Métrica | Reg. Logística | Random Forest | Gradient Boosting | Melhor |
|---------|:-------------:|:-------------:|:-----------------:|:-----:|
| Acurácia | 0,7619 | 0,8299 | **0,8605** | GB |
| Precisão | 0,3736 | 0,4717 | **0,5938** | GB |
| Recall | **0,7234** | 0,5319 | 0,4043 | RL |
| F1-Score | 0,4928 | **0,5000** | 0,4810 | RF |
| ROC-AUC | 0,7954 | 0,7615 | **0,8055** | GB |

A **Regressão Logística** foi o modelo selecionado como vencedor no conjunto de validação (Val F1-Score = 0,5441), demonstrando também a melhor capacidade de identificar funcionários em risco de saída no conjunto de teste final com maior Recall (0,7234).

## Como Reproduzir

### Pré-requisitos
- Python 3.11+
- python3-venv

### Ambiente Virtual (recomendado)

```bash
# Clonar o repositório
git clone https://github.com/Gabriel-Freitas-S/previsao-turnover-funcionarios.git
cd previsao-turnover-funcionarios

# Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Executar o pipeline completo
python3 src/main.py

# Gerar os slides
python3 src/generate_slides.py
```

### Jupyter Notebook

```bash
# Ativar o ambiente (se ainda não estiver ativo)
source .venv/bin/activate

# Iniciar o Jupyter Notebook
python3 -m notebook
```

Abra o notebook em `notebooks/previsao_turnover.ipynb`.

## Estrutura do Projeto

```
previsao-turnover-funcionarios/
├── data/
│   └── HR_Employee_Attrition.csv  # Dataset IBM HR
├── notebooks/
│   └── previsao_turnover.ipynb    # Jupyter Notebook interativo
├── slides/
│   ├── apresentacao_turnover.pdf  # Slides da apresentação (10 páginas)
│   ├── eda_plots.png              # Gráficos da análise exploratória
│   ├── model_comparison.png       # Comparação dos modelos
│   └── metrics.json               # Métricas exportadas pelo main.py
├── src/
│   ├── main.py                 # Pipeline ML completo
│   └── generate_slides.py      # Geração de slides em PDF
├── SPEC/
│   └── Trabalho Machine Learning C3.md  # Especificações do trabalho
├── index.html                  # Página GitHub Pages
├── modelo_turnover.pkl         # Melhor modelo salvo
└── README.md                   # Este arquivo
```

## Tecnologias

- **Python 3.14** — scikit-learn, pandas, matplotlib, seaborn, joblib
- **Editor** — Zed
- **Slides** — fpdf2 (PDF)

## Licença

Projeto acadêmico — Machine Learning, Unidade 3.
