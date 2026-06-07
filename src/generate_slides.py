"""
GERADOR DE SLIDES EM PDF - PREVISAO DE TURNOVER (HR Analytics)
===============================================================
Gera uma apresentacao de 10 slides no formato A4 (210x297 mm)
contendo descricao do problema, metodologia, resultados e conclusoes
do projeto de Machine Learning.

Dataset: HR Analytics - Kaggle (Giri Pujar) | CC0: Public Domain
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
            self.cell(0, 8, "Previsão de Rotatividade de Funcionários - HR Analytics (Kaggle)", align="R", new_x="LMARGIN", new_y="NEXT")
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
        self.cell(0, 8, "Dataset: HR Analytics - Kaggle (Giri Pujar) | CC0: Public Domain", align="C", new_x="LMARGIN", new_y="NEXT")
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

        Args:
            img_path: Caminho absoluto ou relativo para o arquivo de imagem.
            w: Largura da imagem em mm (default: 170).

        Returns:
            True se a imagem foi inserida, False se o arquivo nao existe.
        """
        if os.path.exists(img_path):
            x = (297 - w) / 2
            self.image(img_path, x=x, w=w)
            return True
        return False


def build_presentation() -> None:
    """
    Constroi a apresentacao de 10 slides em formato Paisagem A4,
    alinhada de forma identica (1 para 1) com o Modo Apresentacao do index.html.
    """
    pdf = SlidePDF()
    pdf.alias_nb_pages()

    # =====================================================================
    # Slide 1 - Capa
    # =====================================================================
    pdf.add_title_slide(
        "Previsão de Rotatividade de Funcionários",
        "Classificação Binária para Turnover com HR Analytics (Kaggle)"
    )

    # =====================================================================
    # Slide 2 - 1. Introdução e Relevância do Problema
    # =====================================================================
    pdf.add_section_slide("1", "Introdução e Relevância do Problema")
    pdf.add_body_text(
        "Problema: Predição de saída voluntária de colaboradores (Turnover) usando aprendizado de máquina supervisionado."
    )
    pdf.ln(3)

    pdf.set_fill_color(240, 246, 255)
    pdf.set_draw_color(37, 99, 235)
    pdf.set_line_width(0.4)
    current_y = pdf.get_y()
    pdf.rect(10, current_y, 277, 24, style="FD")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(37, 99, 235)
    pdf.text(13, current_y + 6, "Valor de Negócio (Retenção Estratégica):")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.set_xy(13, current_y + 8)
    pdf.multi_cell(271, 5, "A evasão de funcionários interrompe fluxos operacionais críticos e gera custos de rescisão, recrutamento e treinamento. A predição antecipada permite ao RH agir proativamente antes do pedido de desligamento.")

    pdf.set_xy(10, current_y + 28)
    pdf.add_bullet("Tipo de Problema: Classificação Binária (Classe 1 = Saiu, Classe 0 = Ficou)")
    pdf.add_bullet("Dataset: HR Analytics - Kaggle (Giri Pujar) - 14.999 registros, CC0: Public Domain")
    pdf.add_bullet("Desafio Principal: Desbalanceamento (~76% ficou / ~24% saiu), exigindo métricas adequadas (F1-Score).")

    # =====================================================================
    # Slide 3 - 2. Obtenção e Preparação dos Dados
    # =====================================================================
    pdf.add_section_slide("2", "Obtenção e Preparação dos Dados")
    pdf.add_body_text("Dataset: HR Analytics - Kaggle (Giri Pujar) - 14.999 registros, 10 colunas, CC0: Public Domain.")
    pdf.add_body_text("Colunas originais (traduzidas para PT-BR):")
    pdf.add_bullet("nivel_satisfacao, ultima_avaliacao, numero_projetos, media_horas_mensais")
    pdf.add_bullet("tempo_empresa, acidente_trabalho, promocao_ultimos_5anos, departamento, salario")
    pdf.add_bullet("saiu (variável alvo binária: 0 = ficou, 1 = saiu)")
    pdf.ln(2)
    pdf.add_bullet("Pré-processamento: Sem remoção de colunas (ausência de IDs ou vazamento de dados).")
    pdf.add_bullet("Codificação One-Hot: variáveis categóricas 'departamento' e 'salario'.")
    pdf.add_bullet("Normalização (StandardScaler): features numéricas contínuas.")
    pdf.add_bullet("Particionamento: Divisão estratificada em 60% Treino (8.999), 20% Validação (3.000) e 20% Teste (3.000).")

    # =====================================================================
    # Slide 4 - 3. Análise Exploratória de Dados (EDA)
    # =====================================================================
    pdf.add_section_slide("3", "Análise Exploratória de Dados (EDA)")
    pdf.add_body_text("Principais Insights:")
    pdf.add_bullet("Desbalanceamento moderado: ~76,4% ficou / ~23,6% saiu.")
    pdf.add_bullet("Nível de satisfação muito baixo é o principal preditor de saída.")
    pdf.add_bullet("Funcionários com excesso de horas mensais (>250h) ou carga insuficiente (<150h) saem mais.")
    pdf.add_bullet("Salários baixos e falta de promoção nos últimos 5 anos correlacionam-se com turnover.")
    pdf.add_bullet("Departamentos de Vendas e Técnico concentram mais desligamentos.")
    pdf.ln(2)
    pdf.add_image_centered(os.path.join(SLIDES_DIR, "eda_plots.png"), w=125)

    # =====================================================================
    # Slide 5 - 4. Eixo Analítico Secundário: Bike Sharing
    # =====================================================================
    pdf.add_section_slide("4", "Eixo Analítico Secundário: Bike Sharing")
    pdf.add_body_text(
        "Paradigma de Regressão Contínua: Transposição analítica usando o conjunto de dados da UCI para prever locações acumuladas de bicicletas (cnt)."
    )
    pdf.ln(3)

    pdf.set_fill_color(255, 251, 235)
    pdf.set_draw_color(217, 119, 6)
    pdf.set_line_width(0.4)
    current_y = pdf.get_y()
    pdf.rect(10, current_y, 277, 24, style="FD")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(217, 119, 6)
    pdf.text(13, current_y + 6, "Alerta de Data Leakage (Vazamento de Dados):")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.set_xy(13, current_y + 8)
    pdf.multi_cell(271, 5, "Remoção compulsória das colunas casual e registered. A soma de ambas equivale à variável alvo final. Mantê-las inviabilizaria o modelo preditivo real.")

    pdf.set_xy(10, current_y + 28)
    pdf.add_bullet("Pré-processamento: Exclusão de IDs ordinais (instant) e binarização de variáveis de clima/temporada via One-Hot encoding.")
    pdf.add_bullet("Modelos: Regressão Linear Múltipla (OLS) vs Random Forest Regressor (ensemble não-linear por média de predições).")

    # =====================================================================
    # Slide 6 - 5. Modelos de Classificação e Hiperparâmetros
    # =====================================================================
    pdf.add_section_slide("5", "Modelos de Classificação e Hiperparâmetros")
    pdf.add_bullet("Regressão Logística: Projeta probabilidades usando sigmoide e limites lineares. Otimizado hiperparâmetro de regularização C.")
    pdf.add_bullet("Random Forest Classifier: Ensemble paralelo construído por Bagging de árvores de decisão. Hiperparâmetros: n_estimators e max_depth.")
    pdf.add_bullet("Gradient Boosting Classifier: Treinamento sequencial onde cada árvore corrige os resíduos do conjunto anterior via gradiente.")
    pdf.add_bullet("Estratégia class_weight='balanced': Atribuição de pesos inversamente proporcionais às frequências das classes, corrigindo o desbalanceamento.")
    pdf.ln(3)
    pdf.add_body_text("Otimização: GridSearchCV com StratifiedKFold (5 folds) otimizando F1-Score.")

    # =====================================================================
    # Slide 7 - 6. Modelos de Regressão e Hiperparâmetros
    # =====================================================================
    pdf.add_section_slide("6", "Modelos de Regressão e Hiperparâmetros")
    pdf.add_bullet("Regressão Linear Múltipla: Ajusta uma função de coeficientes lineares tentando encontrar a reta que minimiza a soma dos erros quadráticos.")
    pdf.add_bullet("Random Forest Regressor: Combina múltiplas subárvores de regressão para reduzir a variância e prever o valor contínuo final através da média aritmética das predições de cada árvore.")
    pdf.add_bullet("Otimização por GridSearchCV: Ambos os paradigmas (classificação e regressão) utilizam busca em grade com validação cruzada K-Fold (5 folds) na partição de treino para selecionar os parâmetros ideais.")

    # =====================================================================
    # Slide 8 - 7. Métricas de Avaliação e o Paradoxo da Acurácia
    # =====================================================================
    pdf.add_section_slide("7", "Métricas de Avaliação e o Paradoxo da Acurácia")
    pdf.add_body_text("Métricas de Classificação (Turnover):")
    pdf.add_bullet("Acurácia é enganosa em bases assimétricas.")
    pdf.add_bullet("F1-Score é a métrica principal de seleção (harmonização de Precisão e Recall).")
    pdf.add_bullet("Recall (Sensibilidade) é priorizado para o RH interceptar desligamentos.")
    pdf.ln(2)
    pdf.add_body_text("Métricas de Regressão (Bike Sharing):")
    pdf.add_bullet("RMSE: Penaliza desvios residuais grandes elevados ao quadrado.")
    pdf.add_bullet("MAE: Desvio médio linear tangível das locações.")
    pdf.add_bullet("R² Score: Percentual explicativo da variância.")

    # =====================================================================
    # Slide 9 - 8. Resultados do Modelo de Turnover (Teste)
    # =====================================================================
    pdf.add_section_slide("8", "Resultados do Modelo de Turnover (Teste)")

    current_y = pdf.get_y()

    metrics_path = os.path.join(BASE_DIR, "slides", "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            pass

    def get_val(model_name, metric_name):
        if model_name in metrics and metric_name in metrics[model_name]:
            return f"{metrics[model_name][metric_name]:.4f}".replace(".", ",")
        return "N/A"

    def get_best(metric_name):
        best_val = -1
        best_model = None
        for m in ["RegressaoLogistica", "RandomForest", "GradientBoosting"]:
            if m in metrics and metric_name in metrics[m]:
                val = metrics[m][metric_name]
                if val > best_val:
                    best_val = val
                    best_model = m
        if best_model == "RegressaoLogistica": return "RL"
        elif best_model == "RandomForest": return "RF"
        elif best_model == "GradientBoosting": return "GB"
        return "-"

    data = [
        ["Métrica", "Reg. Logística", "Random Forest", "Grad. Boosting", "Melhor"],
        ["Acurácia", get_val("RegressaoLogistica", "accuracy"), get_val("RandomForest", "accuracy"), get_val("GradientBoosting", "accuracy"), get_best("accuracy")],
        ["Precisão", get_val("RegressaoLogistica", "precision"), get_val("RandomForest", "precision"), get_val("GradientBoosting", "precision"), get_best("precision")],
        ["Recall", get_val("RegressaoLogistica", "recall"), get_val("RandomForest", "recall"), get_val("GradientBoosting", "recall"), get_best("recall")],
        ["F1-Score", get_val("RegressaoLogistica", "f1_score"), get_val("RandomForest", "f1_score"), get_val("GradientBoosting", "f1_score"), get_best("f1_score")],
        ["ROC-AUC", get_val("RegressaoLogistica", "roc_auc"), get_val("RandomForest", "roc_auc"), get_val("GradientBoosting", "roc_auc"), get_best("roc_auc")],
    ]

    col_w = [30, 26, 26, 26, 14]
    pdf.set_y(current_y + 4)
    for i, row in enumerate(data):
        pdf.set_x(10)
        for j, cell in enumerate(row):
            if i == 0:
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_fill_color(25, 55, 109)
                pdf.set_text_color(255, 255, 255)
            elif j == 4:
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(200, 50, 50)
                pdf.set_fill_color(255, 235, 235)
            else:
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(40, 40, 40)
                pdf.set_fill_color(245, 245, 245) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            pdf.cell(col_w[j], 6.5, cell, border=1, fill=True, align="C")
        pdf.ln()

    pdf.set_xy(10, pdf.get_y() + 4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    best_name = metrics.get("best_model", "N/A")
    pdf.multi_cell(122, 4.5, f"Modelo selecionado na validação (melhor Val F1): {best_name}.")

    img_path = os.path.join(SLIDES_DIR, "model_comparison.png")
    if os.path.exists(img_path):
        pdf.image(img_path, x=142, y=current_y + 2, w=145)

    # =====================================================================
    # Slide 10 - 9. Conclusão
    # =====================================================================
    pdf.add_section_slide("9", "Conclusão")
    pdf.add_bullet("O pipeline de Machine Learning demonstrou eficácia na identificação de colaboradores com risco de desligamento usando o dataset HR Analytics (Kaggle).")
    pdf.add_bullet("O nível de satisfação e a média de horas mensais são os atributos mais preditivos de turnover.")
    pdf.add_bullet("Salário baixo e ausência de promoção nos últimos 5 anos amplificam significativamente o risco de saída.")
    pdf.add_body_text("Links do Repositório GitHub e Colab:")
    pdf.set_font("Courier", "", 10)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 6, "  https://github.com/Gabriel-Freitas-S/previsao-turnover-funcionarios", new_x="LMARGIN", new_y="NEXT")
    pdf.output(OUTPUT_PDF)
    print(f"Apresentacao salva em: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_presentation()
