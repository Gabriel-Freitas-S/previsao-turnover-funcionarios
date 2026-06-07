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
        "Classificação Binária para Turnover de RH e Estudo de Caso Bike Sharing"
    )
    # =====================================================================
    # Slide 2 - 1. Introdução e Relevância do Problema
    # =====================================================================
    pdf.add_section_slide("1", "Introdução e Relevância do Problema")
    pdf.add_body_text(
        "Problema: Predição de saída voluntária de colaboradores (Turnover/Attrition) usando aprendizado de máquina supervisionado."
    )
    pdf.ln(3)
    
    # Desenho da Caixa de Destaque (Estilo Highlight Box)
    pdf.set_fill_color(240, 246, 255)
    pdf.set_draw_color(37, 99, 235)
    pdf.set_line_width(0.4)
    current_y = pdf.get_y()
    pdf.rect(10, current_y, 277, 24, style="FD")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(37, 99, 235)
    pdf.text(13, current_y + 6, "Valor de Negócio (Retenção Estratégica e Fluxos Documentais):")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.set_xy(13, current_y + 8)
    pdf.multi_cell(271, 5, "A evasão repentina de funcionários em posições-chave interrompe fluxos operacionais críticos e de faturamento. A predição atua mitigando custos de rescisão, recrutamento e perda do conhecimento institucional.")
    
    pdf.set_xy(10, current_y + 28)
    pdf.add_bullet("Tipo de Problema: Classificação Binária (Classe 1 = Saiu, Classe 0 = Permaneceu)")
    pdf.add_bullet("Desafio Principal: Lidar com o forte desbalanceamento nativo dos dados de RH (apenas ~16% de casos de saída).")

    # =====================================================================
    # Slide 3 - 2. Obtenção e Preparação dos Dados
    # =====================================================================
    pdf.add_section_slide("2", "Obtenção e Preparação dos Dados")
    pdf.add_body_text("Dataset: IBM HR Analytics Employee Attrition & Performance (1470 registros, 35 colunas).")
    pdf.ln(2)
    pdf.add_bullet("Limpeza Crítica: Remoção de atributos sem variância (StandardHours, EmployeeCount, Over18) e crachá (EmployeeNumber).")
    pdf.add_bullet("Transformação de Atributos: Codificação One-Hot para variáveis categóricas qualitativas e Normalização (StandardScaler) para variáveis contínuas salariais e de tempo.")
    pdf.add_bullet("Blindagem contra Vazamento (Data Leakage): Uso de ColumnTransformer e Pipeline ajustados estritamente na base de treino.")
    pdf.add_bullet("Particionamento (SPEC): Divisão estratificada em 60% Treino (882), 20% Validação (294) e 20% Teste (294).")

    # =====================================================================
    # Slide 4 - 3. Análise Exploratória de Dados (EDA)
    # =====================================================================
    pdf.add_section_slide("3", "Análise Exploratória de Dados (EDA)")
    pdf.add_body_text("Mapeamento das Relações Organizacionais:")
    pdf.add_bullet("Forte desbalanceamento nativo: 83,88% Não / 16,12% Sim.")
    pdf.add_bullet("Horas Extras acumulam-se como o maior causador de evasão voluntária por estresse (Burnout).")
    pdf.add_bullet("Salários baixos e início de carreira (poucos anos de casa) impulsionam transições externas.")
    pdf.add_bullet("Vendas é o departamento com maior desgaste relativo.")
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
    
    # Desenho da Caixa de Alerta (Estilo Alert Box)
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
    pdf.add_bullet("Random Forest Classifier: Ensemble paralelo construído por agregação Bootstrap (Bagging) de árvores profundas. Hiperparâmetros: n_estimators e max_depth.")
    pdf.add_bullet("Gradient Boosting Classifier: Treinamento sequencial onde cada árvore corrige os resíduos do conjunto anterior via gradiente.")
    pdf.add_bullet("Estratégia de Custo Classweight: Atribuição de pesos inversamente proporcionais às frequências das classes (class_weight='balanced') na Regressão Logística e Random Forest para calibrar o modelo contra a assimetria.")
    pdf.ln(3)
    pdf.add_body_text("Melhores hiperparâmetros encontrados:")
    pdf.set_font("Courier", "B", 10)
    pdf.set_text_color(25, 55, 109)
    pdf.cell(0, 5, "  Regressão Logística: C=0.1, penalty=L2, solver=lbfgs", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "  Random Forest: n_estimators=200, max_depth=5, min_samples_split=5", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "  Gradient Boosting: learning_rate=0.2, max_depth=3, n_estimators=100", new_x="LMARGIN", new_y="NEXT")

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
    
    # Carrega métricas reais
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
        for m in ["LogisticRegression", "RandomForest", "GradientBoosting"]:
            if m in metrics and metric_name in metrics[m]:
                val = metrics[m][metric_name]
                if val > best_val:
                    best_val = val
                    best_model = m
        if best_model == "LogisticRegression": return "RL"
        elif best_model == "RandomForest": return "RF"
        elif best_model == "GradientBoosting": return "GB"
        return "-"

    data = [
        ["Métrica", "Reg. Logística", "Random Forest", "Grad. Boosting", "Melhor"],
        ["Acurácia", get_val("LogisticRegression", "accuracy"), get_val("RandomForest", "accuracy"), get_val("GradientBoosting", "accuracy"), get_best("accuracy")],
        ["Precisão", get_val("LogisticRegression", "precision"), get_val("RandomForest", "precision"), get_val("GradientBoosting", "precision"), get_best("precision")],
        ["Recall", get_val("LogisticRegression", "recall"), get_val("RandomForest", "recall"), get_val("GradientBoosting", "recall"), get_best("recall")],
        ["F1-Score", get_val("LogisticRegression", "f1_score"), get_val("RandomForest", "f1_score"), get_val("GradientBoosting", "f1_score"), get_best("f1_score")],
        ["ROC-AUC", get_val("LogisticRegression", "roc_auc"), get_val("RandomForest", "roc_auc"), get_val("GradientBoosting", "roc_auc"), get_best("roc_auc")],
    ]
    
    # Tabela à esquerda
    col_w = [30, 26, 26, 26, 14] # total = 122mm
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

    # Legenda da tabela
    pdf.set_xy(10, pdf.get_y() + 4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(122, 4.5, "A Regressão Logística foi selecionada no conjunto de validação (Val F1=0,5441) e provê a melhor sensibilidade operacional (Recall de 72,34%).")

    # Imagem à direita
    img_path = os.path.join(SLIDES_DIR, "model_comparison.png")
    if os.path.exists(img_path):
        pdf.image(img_path, x=142, y=current_y + 2, w=145)

    # =====================================================================
    # Slide 10 - 9. Conclusão
    # =====================================================================
    pdf.add_section_slide("9", "Conclusão")
    pdf.add_bullet("O modelo de Regressão Logística provou ser viável para antecipar evasões de talentos, permitindo ações proativas do RH (Recall = 72,34% no teste).")
    pdf.add_bullet("Horas extras elevadas e remuneração mensal inferior destacam-se como os maiores alavancadores do desgaste corporativo.")
    pdf.add_body_text("Links do Repositório GitHub e Colab:")
    pdf.set_font("Courier", "", 10)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 6, "  https://github.com/Gabriel-Freitas-S/previsao-turnover-funcionarios", new_x="LMARGIN", new_y="NEXT")
    pdf.output(OUTPUT_PDF)
    print(f"Apresentacao salva em: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_presentation()
