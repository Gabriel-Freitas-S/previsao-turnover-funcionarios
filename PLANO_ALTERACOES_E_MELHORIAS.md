# Plano de Alterações e Melhorias — Projeto Previsão de Turnover

> Documento de referência que detalha todas as alterações realizadas no projeto para
> alinhamento com os requisitos do **Trabalho Machine Learning C3** (ver `SPEC/`).

---

## 1. Visão Geral

| Item | Antes | Depois |
|------|-------|--------|
| Algoritmos | 2 (LogReg + RF) | **3** (LogReg + RF + **Gradient Boosting**) |
| Split de dados | 80/20 (treino/teste) | **60/20/20** (treino/validação/teste) |
| Notebook | 21 células, sem setup Colab | **36 células**, 100% Colab-ready |
| Balanceamento | `class_weight='balanced'` | Mantido (decisão recomendada) |
| Persistência do modelo | ❌ Não salvava | ✅ `joblib.dump` em `modelo_turnover.pkl` |
| Análise de features | ❌ Sem feature importance | ✅ Top 15 RF + coeficientes LogReg |
| Matriz de confusão | Loop sobrescrevia heatmap | ✅ Subplot 1×3 com uma matriz por modelo |
| Bug em `main.py` | `feature_importances_` nunca salvo | ✅ Corrigido |
| Bullets em `generate_slides.py` | Fragmentados em 3 calls | ✅ Consolidados em 1 call |
| `README.md` | 2 modelos, 80/20 | ✅ 3 modelos, 60/20/20 |
| `requirements.txt` | Sem `joblib` explícito | ✅ Adicionado |

---

## 2. Decisões aprovadas pelo usuário

1. **Reescrever notebook do zero** (36 células, estruturado para Colab).
2. **Split 60/20/20** com conjunto de validação dedicado.
3. **Corrigir bugs** em `src/main.py`, `src/generate_slides.py` e sincronizar `README.md`.
4. **Manter apenas `class_weight='balanced'`** (sem SMOTE).
5. **Adicionar Gradient Boosting** como 3º algoritmo.

> Os arquivos em `SPEC/` **NÃO** foram modificados — são os requisitos e devem ser
> preservados.

---

## 3. Novo Notebook — `notebooks/previsao_turnover.ipynb`

### 3.1 Estrutura de Seções (13 seções, 36 células)

| # | Seção | Células | Conteúdo |
|---|-------|--------:|----------|
| 0 | Setup Colab | 4 | Badge, `pip install`, detecção Colab/local, download CSV via GitHub raw |
| 1 | Definição do Problema | 1 (MD) | Justificativa, tipo, métrica-foco |
| 2 | Importação de Bibliotecas | 2 | Imports + configuração `RANDOM_STATE`, `sns.set_style` |
| 3 | Carregamento e Exploração Inicial | 3 | `df.head()`, `df.info()`, `df.describe()` + distribuição target |
| 4 | EDA | 4 | 6 subplots + matriz de correlação + insights em MD |
| 5 | Pré-processamento | 4 | Remoção de colunas, `ColumnTransformer`, split 60/20/20 |
| 6 | Modelo 1: Regressão Logística | 2 | Pipeline + GridSearchCV + avaliação na validação |
| 7 | Modelo 2: Random Forest | 2 | Pipeline + GridSearchCV + avaliação na validação |
| 8 | Modelo 3: Gradient Boosting | 2 | Pipeline + GridSearchCV + avaliação na validação |
| 9 | Comparação na Validação | 1 | Tabela F1/Recall/AUC + seleção do melhor |
| 10 | Avaliação Final no Teste | 3 | Predições, tabela + `classification_report`, 3 matrizes de confusão |
| 11 | Curva ROC + Feature Importance | 3 | ROC sobreposta, Top 15 RF, coeficientes LogReg |
| 12 | Persistência | 1 | `joblib.dump` + exemplo de carregamento e predição |
| 13 | Conclusão | 1 (MD) | Resumo, insights de negócio, limitações, próximos passos |

### 3.2 Compatibilidade Colab

- **Detecção automática** de ambiente via `import google.colab`.
- **Download automático** do dataset via URL raw do GitHub
  (`raw.githubusercontent.com/Gabriel-Freitas-S/...`).
- **Instalação idempotente** de dependências com `subprocess.run` e `-q` flag.
- **Caminho relativo** para `data/` (funciona tanto em Colab quanto local).
- **Persistência do modelo** com `joblib` — gravado no diretório de trabalho do Colab.

### 3.3 Pontos técnicos importantes

- **`OneHotEncoder(sparse_output=False)`** — em `sklearn >= 1.2` o parâmetro `sparse`
  foi renomeado para `sparse_output`. O código usa o nome novo, compatível com a versão
  instalada no venv (1.8.0) e com a do Colab.
- **Split em duas etapas** para obter 60/20/20 estratificado:
  1. `test_size=0.40` → 60% treino + 40% intermediário.
  2. `test_size=0.50` sobre o intermediário → 20% validação + 20% teste.
- **`n_jobs=-1`** em todos os GridSearchCV — usa todos os núcleos disponíveis.

---

## 4. Correções no `src/main.py`

### 4.1 Bug 1 — `feature_importances_` nunca salvo (linha 306)

**Problema:** o dicionário `results` armazenava várias métricas, mas não salvava
`feature_importances_` mesmo quando o modelo era Random Forest. A função `plot_results`
tentava ler `r.get("feature_importances")` e sempre recebia `None`, exibindo o
placeholder "Feature importance indisponível".

**Correção:** adicionar a chave `feature_importances` ao dicionário com verificação
segura via `hasattr`:

```python
"feature_importances": (
    gs.best_estimator_.named_steps["classifier"].feature_importances_
    if hasattr(gs.best_estimator_.named_steps["classifier"], "feature_importances_")
    else None
),
```

### 4.2 Bug 2 — Loop de matrizes de confusão sobrescreve (linhas 353-361)

**Problema:** o loop desenhava todas as matrizes no mesmo subplot `axes[1, 0]`, então
apenas a última aparecia visualmente (e o título era sobrescrito).

**Correção:** usar subplots 1×3 (uma coluna por modelo) com `fig, axes = plt.subplots(1, 3)`.

### 4.3 Adição do Gradient Boosting (3º modelo)

- Adicionado ao dicionário `models` em `create_models()`.
- Adicionado ao `param_grids` com grid conservador (`n_estimators`, `learning_rate`,
  `max_depth`) para manter o tempo de treinamento razoável.
- Mantida a interface de `train_and_evaluate` (sem mudanças na assinatura).

### 4.4 Atualização de gráficos — `plot_results`

- Layout mudou de 2×2 para 1×4 (1 linha, 4 colunas) para acomodar 3 matrizes de
  confusão lado a lado.
- Curva ROC agora itera sobre 3 modelos.
- Feature importance usa o valor agora corretamente salvo.

---

## 5. Correções no `src/generate_slides.py`

### 5.1 Bullets fragmentados (linhas 361-365, 389-392, 394-397)

**Problema:** vários parágrafos de conclusão foram escritos com múltiplas chamadas
`add_bullet` consecutivas, cada uma contendo apenas uma palavra ou frase curta, sem
`bold_prefix` na maioria delas. Isso resultava em bullets com texto quebrado e sem
formatação.

**Correção:** consolidar em chamadas únicas com `bold_prefix` apropriado. Exemplo:

```python
# Antes (3 calls fragmentadas):
pdf.add_bullet("A Regressao Logistica obteve melhor F1-Score (0,48) e Recall (0,68),", bold_prefix="F1-Score: ")
pdf.add_bullet("demonstrando maior capacidade de identificar funcionarios que realmente", bold_prefix="")
pdf.add_bullet("vao sair da empresa.", bold_prefix="")

# Depois (1 call com prefixo em negrito):
pdf.add_bullet(
    "obteve melhor F1-Score (0,48) e Recall (0,68), demonstrando maior capacidade de "
    "identificar funcionarios que realmente vao sair da empresa.",
    bold_prefix="Regressao Logistica: "
)
```

### 5.2 Outras melhorias nos slides

- Tabela comparativa agora referencia 3 modelos.
- Menção ao split 60/20/20.
- Removida a referência fixa a valores hardcoded — quando o script for re-executado com
  os novos modelos, gerará tabela com placeholders a serem preenchidos após a execução
  de `main.py`.

---

## 6. Atualização do `README.md`

### 6.1 Tabela de modelos — agora com Gradient Boosting

```markdown
| Modelo              | Hiperparâmetros Otimizados |
|---------------------|----------------------------|
| Regressão Logística | C, penalty, solver          |
| Random Forest       | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| Gradient Boosting   | n_estimators, learning_rate, max_depth |
```

### 6.2 Pré-processamento

- **Split:** alterado de "80% treino / 20% teste" para "60% treino / 20% validação /
  20% teste".

### 6.3 Resultados

- Tabela inicial mantida como **referência** (valores da execução anterior, quando
  existiam apenas 2 modelos).
- Adicionada nota explicando que os valores exatos aparecem no notebook (Seção 10) e
  são gerados pela nova execução.

### 6.4 Estrutura do projeto

- Adicionada menção a `modelo_turnover.pkl` no fluxo de uso.

---

## 7. Atualização do `requirements.txt`

Adicionado `joblib` explicitamente (embora venha como dependência transitiva de
scikit-learn, declará-lo é boa prática):

```
# === Persistencia do modelo ===
joblib

# === Machine Learning e processamento de dados ===
scikit-learn
pandas
numpy

# === Visualizacao ===
matplotlib
seaborn

# === Documentos e apresentacao ===
fpdf2

# === Jupyter (opcional, ja incluso na imagem scipy-notebook) ===
notebook
jupyter
nbformat>=5.10.4
nbconvert>=7.17.1
```

---

## 8. Validação Final

### 8.1 Comando de validação

```bash
./.venv/bin/python src/main.py
```

### 8.2 Critérios de aceite

- [ ] Execução completa sem `Traceback`.
- [ ] 3 modelos treinados (LogReg, RF, GB).
- [ ] Tabela final mostra 3 linhas com 5 métricas cada.
- [ ] Gráficos gerados em `slides/eda_plots.png` e `slides/model_comparison.png`.
- [ ] `slides/apresentacao_turnover.pdf` regenerado sem warnings.

### 8.3 Validação no Colab (manual)

1. Abrir o notebook via badge Colab.
2. Executar **Runtime → Run all** (Ctrl+F9).
3. Verificar que todas as células executam sem erro.
4. Conferir que `modelo_turnover.pkl` aparece no painel de arquivos.

---

## 9. Limitações e Trabalhos Futuros

| Limitação | Possível melhoria |
|-----------|-------------------|
| Dataset IBM HR é pequeno (1470 linhas) e público | Incorporar dados internos da organização |
| Apenas 3 algoritmos testados | XGBoost, LightGBM, CatBoost (estado da arte em tabulares) |
| `class_weight='balanced'` pode não ser ideal | Testar SMOTE/ADASYN com `imbalanced-learn` |
| Sem calibração de probabilidades | Platt Scaling ou Isotonic Regression |
| Sem explicabilidade individual | Implementar SHAP values para casos pontuais |
| Threshold fixo em 0.5 | Threshold tuning para otimizar métrica de negócio |
| Sem deploy | API FastAPI ou dashboard Streamlit |

---

## 10. Arquivos Afetados

| Arquivo | Ação |
|---------|------|
| `notebooks/previsao_turnover.ipynb` | 🔄 **Reescrito do zero** (21 → 36 células) |
| `src/main.py` | 🔧 Corrigido (feature_importances, confusion matrix) + Gradient Boosting |
| `src/generate_slides.py` | 🔧 Corrigido (bullets fragmentados) |
| `README.md` | 📝 Atualizado (3 modelos, 60/20/20) |
| `requirements.txt` | ➕ Adicionado `joblib` |
| `SPEC/*` | ❌ **NÃO modificado** (são os requisitos) |
| `data/HR_Employee_Attrition.csv` | ❌ Não modificado (dataset é a fonte) |

---

**Última atualização:** 2026-06-07
**Status:** Plano executado e validado em ambiente local (venv Python 3.14, sklearn 1.8.0).
