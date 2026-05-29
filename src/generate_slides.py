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

# Diretorios de entrada (imagens) e saida (PDF)
# Os caminhos sao relativos ao diretorio src/
SLIDES_DIR = "../slides"
OUTPUT_PDF = "../slides/apresentacao_turnover.pdf"


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

    def header(self):
        """
        Cabecalho padrao: exibe o nome do projeto em italico no topo
        de todas as paginas a partir da pagina 2 (page_no() > 1).
        """
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "Previsao de Rotatividade de Funcionarios - Machine Learning", align="C")
            self.ln(5)

    def footer(self):
        """
        Rodape padrao: exibe o numero da pagina e o total de paginas
        centralizado no rodape (posicao fixa a 15mm da borda inferior).
        O total de paginas e definido por alias_nb_pages() + {nb}.
        """
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Slide {self.page_no()}/{{nb}}", align="C")

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
        self.ln(50)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(25, 55, 109)
        self.multi_cell(0, 15, title, align="C")
        self.ln(10)
        if subtitle:
            self.set_font("Helvetica", "", 16)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 10, subtitle, align="C")
        self.ln(20)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, "Disciplina: Machine Learning - Unidade 3", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 10, "Dataset: IBM HR Analytics Employee Attrition & Performance", align="C", new_x="LMARGIN", new_y="NEXT")

    def add_section_slide(self, section_num: str, title: str) -> None:
        """
        Cria um slide de secao com numero, titulo e uma linha divisoria
        horizontal na cor azul escuro (#19376D).

        Args:
            section_num: Numero da secao (ex: "1", "5.1", "").
            title: Titulo da secao.
        """
        self.add_page()
        self.ln(40)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(25, 55, 109)
        self.cell(0, 10, f"{section_num}. {title}", align="L")
        self.ln(3)
        self.set_draw_color(25, 55, 109)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)

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
            x = (210 - w) / 2
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

    # =====================================================================
    # Slide 2 - Definicao do Problema
    # =====================================================================
    # Explica o problema de classificacao binaria, as classes envolvidas
    # e a relevancia pratica para o RH corporativo.
    pdf.add_section_slide("1", "Definicao do Problema")
    pdf.add_body_text(
        "Problema: Prever se um funcionario tem alta probabilidade de deixar a empresa "
        "(Turnover/Attrition). Trata-se de um problema de Classificacao Binaria:\n\n"
        "  - Classe Positiva (1): Funcionario saiu da empresa (Yes)\n"
        "  - Classe Negativa (0): Funcionario permaneceu (No)\n\n"
        "Relevancia: A saída repentina de funcionarios que dominam fluxos documentais "
        "criticos gera custos de recrutamento, perda de conhecimento institucional e "
        "riscos a conformidade legal. A retencao preditiva de talentos e uma prioridade "
        "estrategica para RH."
    )

    # =====================================================================
    # Slide 3 - Obtencao e Preparacao dos Dados
    # =====================================================================
    # Descreve o dataset utilizado (origem, tamanho, atributos) e as etapas
    # de pre-processamento (remocao de colunas, codificacao, padronizacao).
    pdf.add_section_slide("2", "Obtencao e Preparacao dos Dados")
    pdf.add_body_text("Dataset: IBM HR Analytics Employee Attrition & Performance (Kaggle)")
    pdf.add_bullet("1470 registros, 35 atributos (23 numericos, 9 categoricos)")
    pdf.add_bullet("0 valores nulos - dados previamente higienizados")
    pdf.add_bullet("Variavel alvo: Attrition (Yes/No)")
    pdf.ln(5)
    pdf.add_body_text("Pre-processamento:")
    pdf.add_bullet("Remocao de colunas sem variancia: EmployeeCount, StandardHours, Over18")
    pdf.add_bullet("Remocao do identificador: EmployeeNumber")
    pdf.add_bullet("Codificacao One-Hot para variaveis categoricas (7 features)")
    pdf.add_bullet("Padronizacao (StandardScaler) para variaveis numericas")
    pdf.add_bullet("ColumnTransformer + Pipeline para evitar data leakage")
    pdf.add_bullet("Divisao: 80% treino / 20% teste (com stratified sampling)")

    # =====================================================================
    # Slide 4 - Analise Exploratoria dos Dados (EDA)
    # =====================================================================
    # Mostra a distribuicao da variavel alvo (desbalanceamento 84/16) e
    # os principais insights extraidos dos graficos. Inclui a imagem com
    # os 6 graficos de EDA.
    pdf.add_section_slide("3", "Analise Exploratoria dos Dados (EDA)")
    pdf.add_body_text("Distribuicao da variavel alvo - Desbalanceamento:")
    pdf.add_bullet("83,88% dos funcionarios permanecem na empresa (No)")
    pdf.add_bullet("16,12% dos funcionarios sairam (Yes)")
    pdf.add_bullet("237 casos positivos vs 1233 casos negativos")
    pdf.ln(3)
    pdf.add_body_text("Principais insights da EDA:")
    pdf.add_bullet("Funcionarios mais jovens apresentam maior propensao a saida")
    pdf.add_bullet("Renda mensal mais baixa correlaciona com maior turnover")
    pdf.add_bullet("Horas extras (OverTime) aumentam significativamente o risco")
    pdf.add_bullet("Departamento de Vendas (Sales) tem maior taxa de attrition")
    pdf.add_bullet("Cargos de Laboratorio Tempo Integral tem maior rotatividade")
    pdf.ln(3)
    pdf.add_image_centered(os.path.join(SLIDES_DIR, "eda_plots.png"), w=175)

    # =====================================================================
    # Slide 5 - Metodologia: Modelos e Otimizacao
    # =====================================================================
    # Apresenta os dois modelos escolhidos (Regressao Logistica e Random
    # Forest), os grids de hiperparametros testados, a estrategia de
    # validacao cruzada e a metrica de otimizacao (F1-Score).
    pdf.add_section_slide("4", "Metodologia - Modelos e Otimizacao")
    pdf.add_body_text("Modelos selecionados:")
    pdf.ln(2)
    pdf.add_bullet("Regressao Logistica (class_weight='balanced') - Modelo linear interpretavel")
    pdf.add_bullet("Random Forest (class_weight='balanced') - Ensemble nao-linear robusto")
    pdf.ln(3)
    pdf.add_body_text("Hiperparametros otimizados via GridSearchCV:")
    pdf.ln(2)
    pdf.add_body_text("Regressao Logistica:")
    pdf.add_bullet("C (regularizacao inversa): [0.01, 0.1, 1, 10, 100]")
    pdf.add_bullet("Penalty: L2 (Ridge)")
    pdf.ln(2)
    pdf.add_body_text("Random Forest:")
    pdf.add_bullet("n_estimators: [100, 200, 300]")
    pdf.add_bullet("max_depth: [5, 10, None]")
    pdf.add_bullet("min_samples_split: [2, 5]")
    pdf.add_bullet("min_samples_leaf: [1, 2]")
    pdf.ln(2)
    pdf.add_body_text("Validacao cruzada: StratifiedKFold (5 folds)\n"
                      "Metrica de otimizacao: F1-Score (classe minoritaria)\n"
                      "Estrategia de balanceamento: class_weight='balanced'")

    # =====================================================================
    # Slide 6 - Resultados e Comparacao de Desempenho
    # =====================================================================
    # Exibe os melhores hiperparametros encontrados pelo GridSearchCV e
    # uma tabela comparativa com 5 metricas para os dois modelos.
    # A tabela destaca o modelo vencedor em cada metrica com fundo
    # levemente rosado e texto em negrito vermelho.
    pdf.add_section_slide("5", "Resultados e Comparacao de Desempenho")
    pdf.add_body_text("Melhores hiperparametros encontrados:")
    pdf.ln(2)
    pdf.set_font("Courier", "B", 10)
    pdf.set_text_color(25, 55, 109)
    pdf.cell(0, 6, "Regressao Logistica:", ln=True)
    pdf.set_font("Courier", "", 9)
    pdf.cell(0, 5, "  C=0.1, penalty=L2, solver=lbfgs", ln=True)
    pdf.ln(3)
    pdf.set_font("Courier", "B", 10)
    pdf.cell(0, 6, "Random Forest:", ln=True)
    pdf.set_font("Courier", "", 9)
    pdf.cell(0, 5, "  n_estimators=100, max_depth=5, min_samples_leaf=2", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Comparacao de Metricas (conjunto de teste):", ln=True)
    pdf.ln(2)

    # Tabela comparativa: 4 colunas (Metrica, RL, RF, Melhor)
    data = [
        ["Metrica", "Reg. Logistica", "Random Forest", "Melhor"],
        ["Acuracia", "0,7619", "0,8231", "RF"],
        ["Precisao", "0,3678", "0,4468", "RF"],
        ["Recall", "0,6809", "0,4468", "RL"],
        ["F1-Score", "0,4776", "0,4468", "RL"],
        ["ROC-AUC", "0,8022", "0,7494", "RL"],
    ]
    col_w = [45, 45, 45, 25]
    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            if i == 0:
                # Linha de cabecalho: fundo azul escuro, texto branco
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_fill_color(25, 55, 109)
                pdf.set_text_color(255, 255, 255)
            elif j == 3:
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

    # =====================================================================
    # Slide 7 - Graficos de Desempenho
    # =====================================================================
    # Slide dedicado exclusivamente ao grafico de comparacao dos modelos
    # (metricas em barras, curva ROC, matrizes de confusao, importancia
    # de features). Mantido em slide separado para melhor legibilidade.
    pdf.add_section_slide("5.1", "Graficos de Desempenho")
    pdf.add_image_centered(os.path.join(SLIDES_DIR, "model_comparison.png"), w=175)
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, "Figura: Comparacao de metricas, curva ROC, matrizes de confusao e importancia de features.")

    # =====================================================================
    # Slide 8 - Discussao dos Resultados
    # =====================================================================
    # Analise critica dos resultados: porque a Regressao Logistica foi
    # escolhida mesmo com acuracia menor, importancia do F1-Score em
    # datasets desbalanceados, e papel do class_weight='balanced'.
    pdf.add_section_slide("6", "Discussao dos Resultados")
    pdf.add_body_text("Analise comparativa:")
    pdf.ln(2)
    pdf.add_bullet("A Regressao Logistica obteve melhor F1-Score (0,48) e Recall (0,68),",
                   bold_prefix="F1-Score: ")
    pdf.add_bullet("demonstrando maior capacidade de identificar funcionarios que realmente",
                   bold_prefix="")
    pdf.add_bullet("vao sair da empresa.", bold_prefix="")
    pdf.ln(2)
    pdf.add_bullet("O Random Forest obteve maior Acuracia (0,82), mas isso e enganoso",
                   bold_prefix="Acuracia: ")
    pdf.add_bullet("devido ao forte desbalanceamento dos dados (84% permanecem).", bold_prefix="")
    pdf.ln(2)
    pdf.add_bullet("A Regressao Logistica e mais indicada para este problema: prioriza",
                   bold_prefix="Conclusao: ")
    pdf.add_bullet("a deteccao de funcionarios em risco (Recall maior), essencial para", bold_prefix="")
    pdf.add_bullet("que o RH possa agir proativamente.", bold_prefix="")
    pdf.ln(3)
    pdf.add_body_text("Importancia do F1-Score como metrica principal:")
    pdf.add_bullet("Acurácia e enganosa em datasets desbalanceados")
    pdf.add_bullet("F1-Score equilibra Precisao e Recall para a classe minoritaria")
    pdf.add_bullet("Class_weight='balanced' foi crucial para melhorar a deteccao da classe positiva")

    # =====================================================================
    # Slide 9 - Conclusoes e Melhorias
    # =====================================================================
    # Resume os principais resultados e lista possiveis melhorias
    # para iteracoes futuras do projeto.
    pdf.add_section_slide("7", "Conclusoes e Melhorias")
    pdf.add_body_text("Conclusoes:")
    pdf.ln(2)
    pdf.add_bullet("O modelo de Regressao Logistica com class_weight='balanced' apresentou o",
                   bold_prefix="")
    pdf.add_bullet("melhor equilibrio entre deteccao de saidas (Recall=0,68) e qualidade", bold_prefix="")
    pdf.add_bullet("das predicoes (F1=0,48).", bold_prefix="")
    pdf.ln(2)
    pdf.add_bullet("Fatores mais relevantes para turnover: horas extras, baixa renda,",
                   bold_prefix="")
    pdf.add_bullet("distancia de casa, poucos anos de empresa e baixa satisfacao", bold_prefix="")
    pdf.add_bullet("com o ambiente de trabalho.", bold_prefix="")
    pdf.ln(3)
    pdf.add_body_text("Possiveis melhorias:")
    pdf.add_bullet("Testar SMOTE como alternativa ao class_weight='balanced'")
    pdf.add_bullet("Explorar outros algoritmos: XGBoost, SVM, Redes Neurais")
    pdf.add_bullet("Coletar mais dados para melhorar a generalizacao")
    pdf.add_bullet("Analise de features importantes para acoes direcionadas de RH")
    pdf.add_bullet("Implementacao de sistema de alerta em tempo real")

    # =====================================================================
    # Slide 10 - Referencias e Repositorio
    # =====================================================================
    # Lista as referencias bibliograficas e o link para o repositorio
    # do projeto no GitHub, alem das tecnologias utilizadas.
    pdf.add_section_slide("", "Referencias e Repositorio")
    pdf.add_body_text("Referencias:")
    pdf.add_bullet("IBM HR Analytics Employee Attrition & Performance - Kaggle")
    pdf.add_bullet("Scikit-learn: Pipeline, GridSearchCV, ColumnTransformer")
    pdf.add_bullet("Documentacao: matplotlib, pandas, seaborn")
    pdf.ln(5)
    pdf.add_body_text("Repositorio do Projeto:")
    pdf.ln(2)
    pdf.set_font("Courier", "", 11)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 8, "https://github.com/Gabriel-Freitas-S/previsao-turnover-funcionarios", ln=True)
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
