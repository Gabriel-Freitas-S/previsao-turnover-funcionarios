"""
GERADOR DE SLIDES EM PDF - PREVISAO DE TURNOVER
=================================================
Gera uma apresentacao de 10 slides no formato A4 (210x297 mm)
contendo descricao do problema, metodologia, resultados e conclusoes
do projeto de Machine Learning.

Dependencias: pip install -r ../requirements.txt
Uso: python3 generate_slides.py
Saida: ../slides/apresentacao_turnover.pdf
"""

from fpdf import FPDF
import os
import json

# Diretorios de entrada (imagens) e saida (PDF)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIDES_DIR = os.path.join(BASE_DIR, "slides")
OUTPUT_PDF = os.path.join(SLIDES_DIR, "apresentacao_turnover.pdf")


class SlidePDF(FPDF):
    """
    Classe personalizada de PDF para a apresentacao de slides.

    Herda de FPDF e sobrescreve os metodos header() e footer()
    para adicionar cabecalho e rodape automaticos em todas as
    paginas (exceto a pagina de capa, que nao tem cabecalho).

    Metodos auxiliares:
        add_title_slide()   - slide de capa com titulo centralizado
        add_section_slide() - slide de secao com numero e titulo
        add_body_text()     - paragrafo de texto corrido
        add_bullet()        - item de lista com marcador
        add_image_centered()- imagem centralizada na pagina
    """

    def __init__(self, **kwargs):
        super().__init__(orientation="landscape", unit="mm", format="A4", **kwargs)

    def header(self):
        """
        Cabecalho padrao: exibe o nome do projeto no topo direito.
        """
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "Previsão de Rotatividade de Funcionários - Machine Learning", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.2)
            self.line(10, 15, 287, 15)
            self.ln(5)

    def footer(self):
        """
        Rodape padrao: exibe o numero da pagina e o total de paginas
        alinhado a direita com uma linha azul de estilo.
        """
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.set_draw_color(25, 55, 109)
        self.set_line_width(0.5)
        self.line(10, 200, 287, 200)
        self.cell(0, 10, f"Slide {self.page_no()}/{{nb}}", align="R")

    def add_title_slide(self, title: str, subtitle: str = "") -> None:
        """
        Cria o slide de capa com titulo e subtitulo centralizados,
        alem de informacoes institucionais (disciplina e dataset).

        A capa usa fonte maior (28pt para o titulo, 16pt para subtitulo)
        e pula aproximadamente 50mm para centralizar verticalmente.

        Args:
            title: Titulo principal do projeto.
            subtitle: Subtitulo opcional (ex: descricao curta).
        """
        self.add_page()
        self.ln(25)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(25, 55, 109)
        self.multi_cell(0, 15, title, align="C")
        self.ln(8)
        if subtitle:
            self.set_font("Helvetica", "", 16)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 10, subtitle, align="C")
        self.ln(10)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Disciplina: Machine Learning - Unidade 3", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "Dataset: IBM HR Analytics Employee Attrition & Performance", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "Integrantes: Gabriel Freitas Souza, Indyanny Rodrigues Peixinho", align="C", new_x="LMARGIN", new_y="NEXT")

    def add_section_slide(self, section_num: str, title: str) -> None:
        """
        Cria um slide de secao com numero, titulo e uma linha divisoria
        horizontal na cor azul escuro (#19376D).

        Args:
            section_num: Numero da secao (ex: "1", "5.1", "").
            title: Titulo da secao.
        """
        self.add_page()
        self.ln(12)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(25, 55, 109)
        prefix = f"{section_num}. " if section_num else ""
        self.cell(0, 10, f"{prefix}{title}", align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_draw_color(25, 55, 109)
        self.line(10, self.get_y(), 287, self.get_y())
        self.ln(6)

    def add_body_text(self, text: str, size: int = 11) -> None:
        """
        Adiciona um paragrafo de texto corrido, resetando a posicao X
        para a margem esquerda antes de escrever.

        Args:
            text: Conteudo textual do paragrafo.
            size: Tamanho da fonte (default: 11pt).
        """
        self.set_font("Helvetica", "", size)
        self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def add_bullet(self, text: str, size: int = 11, bold_prefix: str = "") -> None:
        """
        Adiciona um item de lista com marcador ("- ") e suporte para
        prefixo em negrito.

        Exemplo com bold_prefix:
            add_bullet("texto explicativo", bold_prefix="F1-Score: ")
            Resultado: "- F1-Score: texto explicativo"

        Args:
            text: Texto do item.
            size: Tamanho da fonte (default: 11pt).
            bold_prefix: Texto opcional em negrito antes do texto normal.
        """
        self.set_font("Helvetica", "", size)
        self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin)
        if bold_prefix:
            full = f"-  {bold_prefix}{text}"
        else:
            full = f"-  {text}"
        self.multi_cell(0, 6, full)

    def add_image_centered(self, img_path: str, w: int = 170) -> bool:
        """
        Insere uma imagem centralizada horizontalmente na pagina.

        A largura padrao (170mm) ocupa a maior parte da largura util
        do A4 (210mm - 2*10mm margens = 190mm disponiveis).

        Args:
            img_path: Caminho absoluto ou relativo para o arquivo de imagem.
            w: Largura da imagem em mm (default: 170).

        Returns:
            True se a imagem foi inserida, False se o arquivo nao existe.
        """
        if os.path.exists(img_path):
            # Calcula a posicao X para centralizar: (largura pagina - largura imagem) / 2
            x = (297 - w) / 2
            self.image(img_path, x=x, w=w)
            return True
        return False


def build_presentation() -> None:
    """
    Constroi a apresentacao completa de 10 slides chamando os metodos
    apropriados da classe SlidePDF.

    Estrutura dos slides:
        1. Capa (titulo, subtitulo, disciplina, dataset)
        2. Definicao do Problema (classificacao binaria, relevancia)
        3. Obtencao e Preparacao dos Dados (dataset, pre-processamento)
        4. Analise Exploratoria (distribuicao, insights, grafico EDA)
        5. Metodologia (modelos, grids, validacao cruzada)
        6. Resultados (melhores parametros, tabela comparativa)
        7. Graficos de Desempenho (curva ROC, matrizes, importancias)
        8. Discussao (analise dos resultados, importancia do F1-Score)
        9. Conclusoes e Melhorias (resultados finais, proximos passos)
        10. Referencias e Repositorio (fontes, link do GitHub)
    """
    pdf = SlidePDF()
    pdf.alias_nb_pages()

    # =====================================================================
    # Slide 1 - Capa
    # =====================================================================
    # Titulo principal com subtitulo descritivo e informacoes institucionais.
    pdf.add_title_slide(
        "Previsao de Rotatividade de Funcionarios",
        "Classificacao Binaria para Turnover Corporativo\nIBM HR Analytics Employee Attrition & Performance"
    )

    # Slide 2 - Definicao do Problema
    # =====================================================================
    # Explica o problema de classificacao binaria, as classes envolvidas
    # e a relevancia pratica para o RH corporativo.
    pdf.add_section_slide("1", "Definição do Problema")
    pdf.add_body_text(
        "Problema: Prever se um funcionário tem alta probabilidade de deixar a empresa "
        "(Turnover). Trata-se de um problema de Classificação Binária:\n\n"
        "  - Classe Positiva (1): Funcionário saiu da empresa (Sim)\n"
        "  - Classe Negativa (0): Funcionário permaneceu (Não)\n\n"
        "Relevância: A saída repentina de funcionários que dominam fluxos documentais "
        "críticos gera custos de recrutamento, perda de conhecimento institucional e "
        "riscos à conformidade legal. A retenção preditiva de talentos é uma prioridade "
        "estratégica para o RH."
    )

    # Slide 3 - Obtencao e Preparacao dos Dados
    # =====================================================================
    pdf.add_section_slide("2", "Obtenção e Preparação dos Dados")
    pdf.add_body_text("Dataset: IBM HR Analytics Employee Attrition & Performance (Kaggle)")
    pdf.add_bullet("1470 registros, 35 atributos (23 numéricos, 7 categóricos restantes)")
    pdf.add_bullet("0 valores nulos - dados previamente higienizados e completos")
    pdf.add_bullet("Variável alvo: Turnover (Sim/Não)")
    pdf.ln(5)
    pdf.add_body_text("Pré-processamento:")
    pdf.add_bullet("Remoção de colunas sem variância: ContagemFuncionarios, HorasPadrao, MaiorDe18")
    pdf.add_bullet("Remoção do identificador: NumeroFuncionario")
    pdf.add_bullet("Codificação One-Hot para variáveis categóricas (7 features)")
    pdf.add_bullet("Padronização (StandardScaler) para variáveis numéricas")
    pdf.add_bullet("ColumnTransformer + Pipeline para evitar vazamento de dados (data leakage)")
    pdf.add_bullet("Divisão: 60% treino / 20% validação / 20% teste (estratificada)")

    # =====================================================================
    # Slide 4 - Analise Exploratoria dos Dados (EDA)
    # =====================================================================
    pdf.add_section_slide("3", "Análise Exploratória dos Dados (EDA)")
    pdf.add_body_text("Distribuição da variável alvo - Desbalanceamento:")
    pdf.add_bullet("83,88% dos funcionários permanecem na empresa (Não)")
    pdf.add_bullet("16,12% dos funcionários saíram (Sim)")
    pdf.add_bullet("237 casos positivos vs 1233 casos negativos")
    pdf.ln(3)
    pdf.add_body_text("Principais insights da EDA:")
    pdf.add_bullet("Funcionários mais jovens apresentam maior propensão à saída")
    pdf.add_bullet("Renda mensal mais baixa correlaciona fortemente com maior turnover")
    pdf.add_bullet("Horas extras (HoraExtra) aumentam significativamente o risco")
    pdf.add_bullet("Departamento de Vendas (Sales) tem maior taxa proporcional de turnover")
    pdf.add_bullet("Cargos como Representante de Vendas e Técnico de Laboratório têm maior rotatividade")
    pdf.ln(3)
    pdf.add_image_centered(os.path.join(SLIDES_DIR, "eda_plots.png"), w=140)

    # =====================================================================
    # Slide 5 - Metodologia: Modelos e Otimizacao
    # =====================================================================
    pdf.add_section_slide("4", "Metodologia - Modelos e Otimização")
    pdf.add_body_text("Modelos selecionados:")
    pdf.ln(2)
    pdf.add_bullet("Regressão Logística (class_weight='balanced') - Modelo linear interpretável")
    pdf.add_bullet("Random Forest (class_weight='balanced') - Ensemble não-linear robusto")
    pdf.add_bullet("Gradient Boosting - Ensemble não-linear baseado em boosting")
    pdf.ln(3)
    pdf.add_body_text("Hiperparâmetros otimizados via GridSearchCV:")
    pdf.ln(2)
    pdf.add_body_text("Regressão Logística:")
    pdf.add_bullet("C (regularização inversa): [0.01, 0.1, 1, 10, 100]")
    pdf.add_bullet("Penalty: L2 (Ridge)")
    pdf.ln(2)
    pdf.add_body_text("Random Forest:")
    pdf.add_bullet("n_estimators: [100, 200, 300]")
    pdf.add_bullet("max_depth: [5, 10, None]")
    pdf.add_bullet("min_samples_split: [2, 5]")
    pdf.add_bullet("min_samples_leaf: [1, 2]")
    pdf.ln(2)
    pdf.add_body_text("Gradient Boosting:")
    pdf.add_bullet("n_estimators: [50, 100, 150]")
    pdf.add_bullet("learning_rate: [0.01, 0.1, 0.2]")
    pdf.add_bullet("max_depth: [3, 5]")
    pdf.ln(2)
    pdf.add_body_text("Validação cruzada: StratifiedKFold (5 folds) no Treino\n"
                      "Métrica de otimização: F1-Score (classe minoritária)\n"
                      "Estratégia de balanceamento: class_weight='balanced' (onde aplicável)")

    # =====================================================================
    # Slide 6 - Resultados e Comparação de Desempenho
    # =====================================================================
    # Exibe os melhores hiperparâmetros encontrados pelo GridSearchCV e
    # uma tabela comparativa com 5 métricas para os três modelos.
    pdf.add_section_slide("5", "Resultados e Comparação de Desempenho")
    pdf.add_body_text("Melhores hiperparâmetros encontrados:")
    pdf.ln(2)
    pdf.set_font("Courier", "B", 10)
    pdf.set_text_color(25, 55, 109)
    pdf.cell(0, 6, "Regressão Logística:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 9)
    pdf.cell(0, 5, "  C=0.1, penalty=L2, solver=lbfgs", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Courier", "B", 10)
    pdf.cell(0, 6, "Random Forest:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 9)
    pdf.cell(0, 5, "  n_estimators=200, max_depth=5, min_samples_leaf=1", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Courier", "B", 10)
    pdf.cell(0, 6, "Gradient Boosting:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 9)
    pdf.cell(0, 5, "  learning_rate=0.1, max_depth=3, n_estimators=150", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Comparação de Métricas (conjunto de teste):", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Tenta carregar as métricas reais salvas pelo main.py
    metrics_path = os.path.join(BASE_DIR, "slides", "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            pass

    # Helper para formatar metrica
    def get_val(model_name, metric_name):
        if model_name in metrics and metric_name in metrics[model_name]:
            return f"{metrics[model_name][metric_name]:.4f}".replace(".", ",")
        return "[Aguardando]"

    # Helper para obter o melhor
    def get_best(metric_name):
        best_val = -1
        best_model = None
        for m in ["LogisticRegression", "RandomForest", "GradientBoosting"]:
            if m in metrics and metric_name in metrics[m]:
                val = metrics[m][metric_name]
                if val > best_val:
                    best_val = val
                    best_model = m
        if best_model == "LogisticRegression":
            return "RL"
        elif best_model == "RandomForest":
            return "RF"
        elif best_model == "GradientBoosting":
            return "GB"
        return "-"

    # Tabela comparativa: 5 colunas (Métrica, RL, RF, GB, Melhor)
    data = [
        ["Métrica", "Reg. Logística", "Random Forest", "Grad. Boosting", "Melhor"],
        ["Acurácia", get_val("LogisticRegression", "accuracy"), get_val("RandomForest", "accuracy"), get_val("GradientBoosting", "accuracy"), get_best("accuracy")],
        ["Precisão", get_val("LogisticRegression", "precision"), get_val("RandomForest", "precision"), get_val("GradientBoosting", "precision"), get_best("precision")],
        ["Recall", get_val("LogisticRegression", "recall"), get_val("RandomForest", "recall"), get_val("GradientBoosting", "recall"), get_best("recall")],
        ["F1-Score", get_val("LogisticRegression", "f1_score"), get_val("RandomForest", "f1_score"), get_val("GradientBoosting", "f1_score"), get_best("f1_score")],
        ["ROC-AUC", get_val("LogisticRegression", "roc_auc"), get_val("RandomForest", "roc_auc"), get_val("GradientBoosting", "roc_auc"), get_best("roc_auc")],
    ]
    col_w = [55, 50, 50, 50, 25]
    for i, row in enumerate(data):
        pdf.set_x(33.5)
        for j, cell in enumerate(row):
            if i == 0:
                # Linha de cabecalho: fundo azul escuro, texto branco
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_fill_color(25, 55, 109)
                pdf.set_text_color(255, 255, 255)
            elif j == 4:
                # Coluna "Melhor": destaca o vencedor em vermelho
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(200, 50, 50)
                pdf.set_fill_color(255, 235, 235)
            else:
                # Linhas de dados: intercalacao de cinza claro e branco
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(40, 40, 40)
                pdf.set_fill_color(245, 245, 245) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            pdf.cell(col_w[j], 7, cell, border=1, fill=True, align="C")
        pdf.ln()
    pdf.ln(5)

    # Slide 7 - Gráficos de Desempenho
    # =====================================================================
    pdf.add_section_slide("5.1", "Gráficos de Desempenho")
    pdf.add_image_centered(os.path.join(SLIDES_DIR, "model_comparison.png"), w=140)
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, "Figura: Comparação de métricas, curva ROC, matrizes de confusão e importância de features.")

    # Slide 8 - Discussao dos Resultados
    # =====================================================================
    pdf.add_section_slide("6", "Discussão dos Resultados")
    pdf.add_body_text("Análise comparativa:")
    pdf.ln(2)
    pdf.add_bullet(
        "A Regressão Logística obteve melhor F1-Score (0,48) e Recall (0,72) no teste final, "
        "demonstrando maior capacidade de identificar funcionários que realmente vão sair da empresa.",
        bold_prefix="F1-Score e Recall: "
    )
    pdf.ln(2)
    pdf.add_bullet(
        "O Random Forest e Gradient Boosting obtiveram maior Acurácia (0,81 e 0,86), mas recall inferior, "
        "o que é enganoso devido ao forte desbalanceamento dos dados (84% permanecem).",
        bold_prefix="Acurácia: "
    )
    pdf.ln(2)
    pdf.add_bullet(
        "A Regressão Logística é mais indicada para este problema: prioriza a detecção de funcionários "
        "em risco (Recall maior de 0,72), essencial para que o RH possa agir proativamente.",
        bold_prefix="Conclusão: "
    )
    pdf.ln(3)
    pdf.add_body_text("Importância do F1-Score como métrica principal:")
    pdf.add_bullet("Acurácia é enganosa em datasets desbalanceados (um modelo que chute 'não' teria 84% de acurácia)")
    pdf.add_bullet("F1-Score equilibra Precisão e Recall para a classe minoritária")
    pdf.add_bullet("Class_weight='balanced' foi crucial para melhorar a detecção da classe positiva")

    # Slide 9 - Conclusoes e Melhorias
    # =====================================================================
    pdf.add_section_slide("7", "Conclusões e Melhorias")
    pdf.add_body_text("Conclusões:")
    pdf.ln(2)
    pdf.add_bullet(
        "O modelo de Regressão Logística com class_weight='balanced' apresentou o melhor equilíbrio "
        "entre detecção de saídas (Recall=0,72) e qualidade das predições (F1=0,48) no conjunto de teste.",
        bold_prefix="Melhor Modelo: "
    )
    pdf.ln(2)
    pdf.add_bullet(
        "Fatores mais relevantes para turnover: horas extras (HoraExtra), baixa renda (RendaMensal), "
        "distância de casa (DistanciaTrabalho), poucos anos de empresa e satisfação no trabalho.",
        bold_prefix="Fatores de Risco: "
    )
    pdf.ln(3)
    pdf.add_body_text("Possíveis melhorias:")
    pdf.add_bullet("Testar SMOTE como alternativa ao class_weight='balanced'")
    pdf.add_bullet("Explorar outros algoritmos: XGBoost, LightGBM, SVM, Redes Neurais")
    pdf.add_bullet("Coletar mais dados para melhorar a generalização")
    pdf.add_bullet("Análise de features importantes para ações direcionadas de RH")
    pdf.add_bullet("Implementação de sistema de alerta em tempo real")

    # =====================================================================
    # Slide 10 - Referencias e Repositorio
    # =====================================================================
    pdf.add_section_slide("", "Referências e Repositório")
    pdf.add_body_text("Referências:")
    pdf.add_bullet("IBM HR Analytics Employee Attrition & Performance - Kaggle")
    pdf.add_bullet("Scikit-learn: Pipeline, GridSearchCV, ColumnTransformer")
    pdf.add_bullet("Documentação: matplotlib, pandas, seaborn")
    pdf.ln(5)
    pdf.add_body_text("Repositório do Projeto:")
    pdf.ln(2)
    pdf.set_font("Courier", "", 11)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 8, "https://github.com/Gabriel-Freitas-S/previsao-turnover-funcionarios", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 6, "Tecnologias: Python, scikit-learn, pandas, matplotlib, seaborn\n"
                         "Container: jupyter/scipy-notebook via Podman\n"
                         "Editor: Zed com Dev Containers")

    # Gera o arquivo PDF final
    pdf.output(OUTPUT_PDF)
    print(f"Apresentacao salva em: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_presentation()
