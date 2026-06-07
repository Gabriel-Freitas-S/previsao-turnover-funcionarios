# Previsão de Rotatividade de Funcionários (Turnover)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gabriel-Freitas-S/previsao-turnover-funcionarios/blob/main/notebooks/previsao_turnover.ipynb)

Projeto de **Machine Learning** para previsão de rotatividade (*turnover*) de funcionários usando o dataset **HR Analytics (Giri Pujar, Kaggle)**.

**Integrantes:** Gabriel Freitas Souza & Indyanny Rodrigues Peixinho  
**Disciplina:** Machine Learning — Unidade 3

---

## 📊 Dataset

| Atributo | Valor |
|---|---|
| **Nome** | HR Analytics |
| **Autor** | Giri Pujar |
| **Fonte** | [Kaggle](https://www.kaggle.com/datasets/giripujar/hr-analytics) |
| **Licença** | CC0: Public Domain |
| **Registros** | 14.999 |
| **Colunas** | 10 |
| **Variável Alvo** | `saiu` (0 = ficou, 1 = saiu) |
| **Taxa de Turnover** | ~23,6% |

### Colunas (traduzidas para PT-BR)

| Coluna | Original | Tipo | Descrição |
|---|---|---|---|
| `nivel_satisfacao` | satisfaction_level | Float [0–1] | Nível de satisfação do funcionário |
| `ultima_avaliacao` | last_evaluation | Float [0–1] | Pontuação da última avaliação de desempenho |
| `numero_projetos` | number_project | Inteiro | Quantidade de projetos atribuídos |
| `media_horas_mensais` | average_montly_hours | Inteiro | Média de horas trabalhadas por mês |
| `tempo_empresa` | time_spend_company | Inteiro | Anos de empresa |
| `acidente_trabalho` | Work_accident | Binário (0/1) | Sofreu acidente de trabalho |
| **`saiu` (alvo)** | left | **Binário (0/1)** | **Saiu da empresa** |
| `promocao_ultimos_5anos` | promotion_last_5years | Binário (0/1) | Recebeu promoção nos últimos 5 anos |
| `departamento` | Department / sales | Categórico | Departamento do funcionário |
| `salario` | salary | Categórico | Faixa salarial: baixo, medio, alto |

---

## 🤖 Modelos e Resultados

Três classificadores treinados com **GridSearchCV + StratifiedKFold (5 folds)** otimizando **F1-Score**:

| Métrica | Regressão Logística | Random Forest | Gradient Boosting |
|---|---|---|---|
| Acurácia | 98,40% | 98,77% | 98,60% |
| Precisão | 95,08% | **97,72%** | 97,57% |
| Recall | **98,31%** | 97,03% | 96,47% |
| **F1-Score** ★ | 96,67% | **97,38%** | 97,02% |
| ROC-AUC | 99,89% | 99,86% | **99,92%** |

> ★ Métrica principal de seleção (desbalanceamento ~76% ficou / ~24% saiu)

### 🏆 Modelo Selecionado: **Gradient Boosting**
Selecionado pelo maior **F1-Score no conjunto de validação (Val F1 = 0,9695)**.

**Melhores Hiperparâmetros (GridSearchCV):**
- `Regressão Logística`: C=0.01 | penalty=l2 | solver=lbfgs
- `Random Forest`: n_estimators=100 | max_depth=None | min_samples_split=2 | min_samples_leaf=1
- `Gradient Boosting`: learning_rate=0.1 | max_depth=3 | n_estimators=100

---

## 📂 Estrutura do Projeto

```
previsao-turnover-funcionarios/
│
├── data/
│   ├── HR_Analytics.csv          # Dataset principal (14.999 registros, PT-BR)
│   └── HR_Dados_Limpos.csv       # Dataset anterior (legado)
│
├── notebooks/
│   └── previsao_turnover.ipynb   # Notebook completo com análise e modelagem
│
├── slides/
│   ├── eda_plots.png             # Gráficos de EDA gerados pelo main.py
│   ├── model_comparison.png      # Comparação de modelos gerada pelo main.py
│   ├── metrics.json              # Métricas reais do último treinamento
│   └── apresentacao_turnover.pdf # Apresentação PDF de 10 slides (A4 Paisagem)
│
├── src/
│   ├── main.py                   # Pipeline de ML: treino, avaliação e EDA
│   └── generate_slides.py        # Gerador de apresentação PDF
│
├── index.html                    # Interface web com visualização e modo slideshow
├── modelo_turnover.pkl           # Melhor modelo persistido (GradientBoosting)
├── requirements.txt              # Dependências do projeto
└── README.md                     # Este arquivo
```

---

## 🚀 Como Executar

### Pré-requisitos

```bash
# Criar e ativar o virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### Executar o Pipeline de ML

```bash
# Treina os modelos, gera gráficos EDA e de comparação, exporta métricas e o pkl
python3 src/main.py
```

### Gerar Apresentação PDF

```bash
# Gera slides/apresentacao_turnover.pdf (10 slides, A4 Paisagem)
python3 src/generate_slides.py
```

### Abrir Interface Web

```bash
# Abrir index.html no navegador (carrega métricas automaticamente do metrics.json)
xdg-open index.html
```

### Executar Notebook

```bash
# Localmente
jupyter notebook notebooks/previsao_turnover.ipynb

# Ou abrir direto no Google Colab
# https://colab.research.google.com/github/Gabriel-Freitas-S/previsao-turnover-funcionarios/blob/main/notebooks/previsao_turnover.ipynb
```

---

## 🔍 Principais Insights (EDA)

- **Satisfação** (`nivel_satisfacao` < 0,35): Fator mais preditivo de saída voluntária
- **Carga horária**: Excesso (>250h/mês) ou subcarga (<150h) eleva a taxa de turnover
- **Salário baixo** (`salario = baixo`): Maior concentração de desligamentos
- **Sem promoção** nos últimos 5 anos: Amplifica significativamente o risco
- **Departamentos** Vendas e Técnico: Maiores índices de rotatividade

---

## 📈 Divisão dos Dados (60 / 20 / 20 — Estratificada)

| Partição | Registros | % Turnover |
|---|---|---|
| Treino | 8.999 | ~23,6% |
| Validação | 3.000 | ~23,6% |
| Teste Final | 3.000 | ~23,6% |

> Particionamento estratificado garante a proporção de turnover em todas as partições.  
> O `fit` do preprocessador ocorre **somente** no treino (prevenção de Data Leakage).

---

## 📦 Dependências

```
pandas
numpy
scikit-learn
matplotlib
seaborn
fpdf2
joblib
nbformat
```

---

## 🔗 Links

- **Kaggle Dataset:** https://www.kaggle.com/datasets/giripujar/hr-analytics
- **Google Colab:** https://colab.research.google.com/github/Gabriel-Freitas-S/previsao-turnover-funcionarios/blob/main/notebooks/previsao_turnover.ipynb
- **GitHub:** https://github.com/Gabriel-Freitas-S/previsao-turnover-funcionarios
