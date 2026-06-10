"""
================================================================================
SCRIPT ACADÊMICO: GERADOR AUTOMÁTICO DE SLIDES EM PDF (A4 PAISAGEM)
================================================================================
Disciplina: Machine Learning / Apresentação de Trabalho Científico
Objetivo:
    Gerar de forma programática um arquivo PDF contendo a apresentação de slides
    do projeto de previsão de turnover (rotatividade). A apresentação é estruturada
    em 10 slides no formato A4 Paisagem (Landscape), alinhada diretamente com
    o modo slideshow interativo da interface web (index.html).

Biblioteca Utilizada:
    FPDF (fpdf2) - Biblioteca Python para geração de PDFs dinâmica.
    Permite controle de fontes, cores, formas geométricas e posicionamento
    preciso das imagens geradas pelo pipeline de Machine Learning (main.py).

Estrutura da Apresentação de Slides:
    Slide 1: Capa (Título, Autores, Disciplina, Contexto)
    Slide 2: 1. Introdução e Relevância do Problema (Evasão voluntária no RH)
    Slide 3: 2. Obtenção e Preparação dos Dados (Limpeza, escala, One-Hot)
    Slide 4: 3. Análise Exploratória de Dados (EDA) (Visualizações e Insights)
    Slide 5: 4. Eixo Analítico Secundário (Estudo Bike Sharing UCI)
    Slide 6: 5. Modelos de Classificação e Hiperparâmetros (Reg. Logística, RF, GB)
    Slide 7: 6. Modelos de Regressão e Hiperparâmetros (Reg. Linear, RF Regressor)
    Slide 8: 7. Métricas de Avaliação e o Paradoxo da Acurácia (F1, Recall, RMSE)
    Slide 9: 8. Resultados do Modelo de Turnover (Tabela de Métricas e Curva ROC)
    Slide 10: 9. Conclusão e Encerramento
================================================================================
"""

from fpdf import FPDF
import os
import json

# Definição das pastas de entrada (gráficos gerados) e saída (PDF final)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIDES_DIR = os.path.join(BASE_DIR, "slides")
OUTPUT_PDF = os.path.join(SLIDES_DIR, "apresentacao_turnover.pdf")


class SlidePDF(FPDF):
    """
    Classe estendida da FPDF customizada para formato de apresentação de slides.
    
    Ajustamos a orientação padrão para "landscape" (paisagem) e o formato
    para "A4", resultando em uma área útil de trabalho de 297 mm de largura
    por 210 mm de altura. Sobrescrevemos os métodos nativos de cabeçalho e rodapé.
    """

    def __init__(self, **kwargs):
        super().__init__(orientation="landscape", unit="mm", format="A4", **kwargs)

    def header(self):
        """
        Gera automaticamente o cabeçalho em todos os slides (exceto na capa).
        
        Desenha o título curto do projeto e uma linha cinza de divisão.
        """
        # A capa (página 1) não deve exibir o cabeçalho padrão
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)  # Fonte Helvetica em Itálico (9pt)
            self.set_text_color(100, 100, 100) # Cor cinza médio
            # Escreve o texto alinhado à direita
            self.cell(0, 8, "Previsão de Rotatividade de Funcionários - HR Analytics (Kaggle)", align="R", new_x="LMARGIN", new_y="NEXT")
            
            # Desenha uma linha horizontal sutil para separar o cabeçalho do conteúdo
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.2)
            self.line(10, 15, 287, 15)  # Coordenadas X1, Y1, X2, Y2 em milímetros
            self.ln(5)

    def footer(self):
        """
        Gera automaticamente o rodapé em todos os slides da apresentação.
        
        Adiciona uma linha divisória azul escuro (tom institucional) e
        imprime o número do slide corrente sobre o total de slides (ex: 'Slide 2/10').
        """
        self.set_y(-15)  # Posiciona o cursor 15 mm antes do final da página
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        
        # Desenha a barra azul escuro decorativa do rodapé
        self.set_draw_color(25, 55, 109)
        self.set_line_width(0.5)
        self.line(10, 200, 287, 200)
        
        # O marcador {nb} é substituído automaticamente pelo número total de páginas do PDF
        self.cell(0, 10, f"Slide {self.page_no()}/{{nb}}", align="R")

    def add_title_slide(self, title: str, subtitle: str = "") -> None:
        """
        Constrói o slide de abertura (capa) com título, subtítulo e metadados.
        """
        self.add_page()
        self.ln(25)  # Avança o cursor para baixo para centralizar verticalmente
        
        # Título principal em negrito destacado
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(25, 55, 109)  # Azul escuro institucional
        self.multi_cell(0, 15, title, align="C")
        self.ln(8)
        
        # Subtítulo (se fornecido) em cinza escuro
        if subtitle:
            self.set_font("Helvetica", "", 16)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 10, subtitle, align="C")
        self.ln(10)
        
        # Informações institucionais adicionadas na parte inferior da capa
        self.set_font("Helvetica", "", 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Disciplina: Machine Learning - Unidade 3", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "Dataset: HR Analytics - Kaggle (Giri Pujar) | CC0: Public Domain", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "Integrantes: Gabriel Freitas Souza, Indyanny Rodrigues Peixinho", align="C", new_x="LMARGIN", new_y="NEXT")

    def add_section_slide(self, section_num: str, title: str) -> None:
        """
        Cria um slide de conteúdo comum contendo o título da seção atual.
        
        Args:
            section_num: String representando a numeração da seção (ex: "1", "2").
            title: Título descritivo do slide.
        """
        self.add_page()
        self.ln(12)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(25, 55, 109)
        prefix = f"{section_num}. " if section_num else ""
        self.cell(0, 10, f"{prefix}{title}", align="L", new_x="LMARGIN", new_y="NEXT")
        
        # Linha azul escura de demarcação do título do slide
        self.ln(2)
        self.set_draw_color(25, 55, 109)
        self.line(10, self.get_y(), 287, self.get_y())
        self.ln(6)

    def add_body_text(self, text: str, size: int = 11) -> None:
        """
        Escreve um parágrafo de texto explicativo corrido.
        """
        self.set_font("Helvetica", "", size)
        self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def add_bullet(self, text: str, size: int = 11, bold_prefix: str = "") -> None:
        """
        Adiciona um item de lista (bullet point) com marcador clássico "- ".
        
        Args:
            text: Conteúdo textual do item da lista.
            size: Tamanho da fonte em pontos.
            bold_prefix: Prefixo em negrito opcional para destacar termos chave.
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
        Insere uma imagem centralizada na página horizontalmente.
        
        Args:
            img_path: Caminho completo para a imagem PNG ou JPG.
            w: Largura desejada em milímetros para a imagem.
            
        Returns:
            Boolean indicando se a imagem existia e foi carregada com sucesso.
        """
        if os.path.exists(img_path):
            x = (297 - w) / 2  # Calcula a coordenada X para centralizar em relação aos 297 mm
            self.image(img_path, x=x, w=w)
            return True
        return False


def build_presentation() -> None:
    """
    Função principal que orquestra a montagem do PDF slide por slide.
    
    Toda a estrutura textual é idêntica à visualizada na aplicação web (index.html).
    """
    pdf = SlidePDF()
    # Habilita a contagem total de páginas automática
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

    # Cria uma caixa decorativa destacada (Callout Box) em tom azul claro
    pdf.set_fill_color(240, 246, 255)
    pdf.set_draw_color(37, 99, 235)
    pdf.set_line_width(0.4)
    current_y = pdf.get_y()
    pdf.rect(10, current_y, 277, 24, style="FD")  # FD = Fill and Draw (preenche e contorna)
    
    # Texto da caixa
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(37, 99, 235)
    pdf.text(13, current_y + 6, "Valor de Negócio (Retenção Estratégica):")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.set_xy(13, current_y + 8)
    pdf.multi_cell(271, 5, "A evasão de funcionários interrompe fluxos operacionais críticos e gera custos de rescisão, recrutamento e treinamento. A predição antecipada permite ao RH agir proativamente antes do pedido de desligamento.")

    # Restaura posição após a caixa
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
    # Insere o gráfico combinado de EDA gerado pelo main.py
    pdf.add_image_centered(os.path.join(SLIDES_DIR, "eda_plots.png"), w=125)

    # =====================================================================
    # Slide 5 - 4. Eixo Analítico Secundário: Bike Sharing
    # =====================================================================
    pdf.add_section_slide("4", "Eixo Analítico Secundário: Bike Sharing")
    pdf.add_body_text(
        "Paradigma de Regressão Contínua: Transposição analítica usando o conjunto de dados da UCI para prever locações acumuladas de bicicletas (cnt)."
    )
    pdf.ln(3)

    # Caixa de alerta decorativo em tom amarelo/laranja destacando Data Leakage
    pdf.set_fill_color(255, 251, 235)
    pdf.set_draw_color(217, 119, 6)
    pdf.set_line_width(0.4)
    current_y = pdf.get_y()
    pdf.rect(10, current_y, 277, 24, style="FD")
    
    # Texto do alerta
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(217, 119, 6)
    pdf.text(13, current_y + 6, "Alerta de Data Leakage (Vazamento de Dados):")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.set_xy(13, current_y + 8)
    pdf.multi_cell(271, 5, "Remoção compulsória das colunas casual e registered. A soma de ambas equivale à variável alvo final. Mantê-las inviabilizaria o modelo preditivo real no mundo real.")

    # Restaura posição
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
    pdf.add_bullet("RMSE (Root Mean Squared Error): Penaliza desvios residuais grandes elevados ao quadrado.")
    pdf.add_bullet("MAE (Mean Absolute Error): Desvio médio linear tangível das locações.")
    pdf.add_bullet("R² Score (Coeficiente de Determinação): Percentual explicativo da variância.")

    # =====================================================================
    # Slide 9 - 8. Resultados do Modelo de Turnover (Teste)
    # =====================================================================
    pdf.add_section_slide("8", "Resultados do Modelo de Turnover (Teste)")

    current_y = pdf.get_y()

    # Carrega métricas calculadas no treinamento para popular a tabela dinamicamente
    metrics_path = os.path.join(BASE_DIR, "slides", "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            pass

    # Funções para buscar os valores reais do JSON e formatar
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

    # Dados estruturados da tabela
    data = [
        ["Métrica", "Reg. Logística", "Random Forest", "Grad. Boosting", "Melhor"],
        ["Acurácia", get_val("RegressaoLogistica", "accuracy"), get_val("RandomForest", "accuracy"), get_val("GradientBoosting", "accuracy"), get_best("accuracy")],
        ["Precisão", get_val("RegressaoLogistica", "precision"), get_val("RandomForest", "precision"), get_val("GradientBoosting", "precision"), get_best("precision")],
        ["Recall", get_val("RegressaoLogistica", "recall"), get_val("RandomForest", "recall"), get_val("GradientBoosting", "recall"), get_best("recall")],
        ["F1-Score", get_val("RegressaoLogistica", "f1_score"), get_val("RandomForest", "f1_score"), get_val("GradientBoosting", "f1_score"), get_best("f1_score")],
        ["ROC-AUC", get_val("RegressaoLogistica", "roc_auc"), get_val("RandomForest", "roc_auc"), get_val("GradientBoosting", "roc_auc"), get_best("roc_auc")],
    ]

    # Desenho da tabela no PDF usando FPDF
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
                # Alterna cores de linhas para leitura facilitada
                pdf.set_fill_color(245, 245, 245) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            pdf.cell(col_w[j], 6.5, cell, border=1, fill=True, align="C")
        pdf.ln()

    # Legenda da tabela
    pdf.set_xy(10, pdf.get_y() + 4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    best_name = metrics.get("best_model", "N/A")
    pdf.multi_cell(122, 4.5, f"Modelo selecionado na validação (melhor Val F1): {best_name}.")

    # Insere o gráfico de comparação gerado pelo main.py no lado direito do slide
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
    pdf.add_body_text("Links do Repositório GitHub:")
    pdf.set_font("Courier", "", 10)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 6, "  https://github.com/Gabriel-Freitas-S/previsao-turnover-funcionarios", new_x="LMARGIN", new_y="NEXT")
    
    # Salva o arquivo PDF gerado
    pdf.output(OUTPUT_PDF)
    print(f"Apresentação PDF gerada com sucesso e salva em: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_presentation()
