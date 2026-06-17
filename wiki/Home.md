# Wiki do Projeto: Previsão de Rotatividade de Funcionários (Turnover)

Bem-vindo à wiki oficial do projeto de **Previsão de Rotatividade de Funcionários (Turnover)**. Este projeto foi desenvolvido como parte da disciplina de **Machine Learning (Unidade 3)**.

---

## 👥 Equipe
* **Gabriel Freitas Souza**
* **Indyanny Rodrigues Peixinho**

---

## 🎯 Objetivo Geral
O objetivo principal é conceber e implementar um pipeline completo de Machine Learning para prever se um funcionário possui propensão a deixar a empresa (Turnover). A predição antecipada é uma ferramenta estratégica crucial para o setor de Recursos Humanos (RH), permitindo intervenções proativas de retenção de talentos antes que ocorra a rescisão contratual.



---

## 📁 Páginas da Wiki
Para explorar os detalhes do projeto, navegue pelas seções estruturadas abaixo:

*   📊 **[1. Dataset & Análise Exploratória (EDA)](1.-Dataset-&-Análise-Exploratória-de-Dados-\(EDA\))**
    *   *Detalhamento dos atributos do dataset HR Analytics e os principais insights gerados sobre a rotatividade dos colaboradores.*
*   ⚙️ **[2. Pré-processamento e Divisão dos Dados](2.-Pré-processamento-e-Divisão-dos-Dados)**
    *   *Estratégias de tratamento de dados (One-Hot Encoding, StandardScaler), particionamento estratificado e prevenção de Data Leakage.*
*   🤖 **[3. Modelos de Machine Learning](3.-Modelos-de-Machine-Learning)**
    *   *Detalhes sobre os classificadores testados (Regressão Logística, Random Forest e Gradient Boosting), otimização de hiperparâmetros com GridSearchCV e balanceamento de classes.*
*   📈 **[4. Avaliação e Resultados](4.-Avaliação-e-Resultados)**
    *   *Comparação de desempenho dos modelos, curva ROC, matrizes de confusão e a importância do F1-Score diante do Paradoxo da Acurácia.*
*   🌐 **[5. Interface Web e Pyodide](5.-Interface-Web-e-Pyodide)**
    *   *Funcionamento do painel interativo frontend executando o modelo diretamente no navegador do cliente usando Pyodide (WASM).*
*   📁 **[6. Estrutura e Explicação do Código](6.-Estrutura-e-Explicação-do-Código)**
    *   *Detalhamento técnico passo a passo e documentação de funções de cada arquivo do projeto: [main.py](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/src/main.py), [generate_slides.py](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/src/generate_slides.py), [export_pyodide_model.py](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/src/export_pyodide_model.py) e [index.html](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/index.html).*
*   🚀 **[7. Como Executar e Reproduzir](7.-Como-Executar-e-Reproduzir)**
    *   *Guia prático de instalação de dependências, execução do pipeline de ML, geração de slides em PDF e inicialização da interface web.*

---

## 🗂 Estrutura do Repositório
A organização dos arquivos no repositório segue a estrutura abaixo:
* [data/](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/data): Contém os conjuntos de dados utilizados (*HR_Analytics.csv*).
* [notebooks/](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/notebooks): Contém o notebook Jupyter experimental [previsao_turnover.ipynb](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/notebooks/previsao_turnover.ipynb).
* [slides/](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/slides): Armazena os gráficos exportados pelo pipeline e a apresentação final gerada em PDF.
* [src/](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/src): Contém os scripts fonte em Python contendo o pipeline de ML, exportador Pyodide e o gerador de slides.
* [index.html](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/index.html): Interface web interativa (dashboard de resultados + simulador em tempo real).
* [modelo_turnover.pkl](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/modelo_turnover.pkl): Melhor modelo serializado pronto para produção.
* [modelo_turnover_pyodide.pkl](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/modelo_turnover_pyodide.pkl): Modelo exportado em formato simplificado para execução em Pyodide.
* [requirements.txt](file:///home/gabriel-freitas-souza/Projetos/previsao-turnover-funcionarios/requirements.txt): Lista de dependências Python do projeto.
