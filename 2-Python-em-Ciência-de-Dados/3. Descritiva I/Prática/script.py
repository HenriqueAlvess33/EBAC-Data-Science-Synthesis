import io
import time
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Criação de um identificador para bons pagadores e maus pagadores",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="varig_icon.png",
)

with st.spinner("Carregando base de dados"):
    df = pd.read_csv("./Dataset/SINASC_RO_2019.csv", keep_default_na=True)


def plotagem_de_graficos_contagem(graph_type, df=df):

    if graph_type == "Primário":
        eixo_x_gestacao = ["37 a 41", "32 a 36", "42+", "28 a 31", "22 a 27", "-22"]

        plt.close("all")
        plt.rc("figure", figsize=(15, 13))
        fig, axes = plt.subplots(2, 3)
        plt.subplots_adjust(wspace=0.25, hspace=0.25)

        sns.countplot(ax=axes[0, 0], data=df, x="APGAR1")
        sns.countplot(ax=axes[0, 1], data=df, x="APGAR5")
        sns.countplot(ax=axes[0, 2], data=df, x="SEXO")
        sns.countplot(ax=axes[1, 1], data=df, x="ESTCIVMAE")
        ax = sns.countplot(ax=axes[1, 0], data=df, x="GESTACAO")

        labels = ax.set_xticklabels(eixo_x_gestacao)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, width=800)

    elif graph_type == "Secundário":
        eixo_x_gestacao = ["37 a 41", "32 a 36", "42+", "28 a 31", "22 a 27", "-22"]
        plt.close("all")
        plt.rc("figure", figsize=(15, 13))
        fig, axes = plt.subplots(2, 2)

        ax = sns.countplot(ax=axes[0, 0], data=df, x="SEXO")
        ax.set_ylabel("Frequência de genêros")

        ax = sns.countplot(ax=axes[0, 1], data=df, x="ESTCIVMAE")
        ax.set_ylabel("Frequência de estado civíl da mãe")

        ax = sns.countplot(ax=axes[1, 0], data=df, x="GESTACAO")
        ax.set_ylabel("Frequência do tempo de gestação")
        labels = ax.set_xticklabels(eixo_x_gestacao)

        ax = sns.countplot(ax=axes[1, 1], data=df, x="CONSULTAS")
        ax.set_ylabel("Frequência de consultas")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, width=800)

    elif graph_type == "Boxplot":
        ylabels = {"IDADEMAE": "Idade da mãe", "PESO": "Peso do bebê"}
        variavel = st.selectbox(
            "Selecione a variável a ser visualizada", ["IDADEMAE", "PESO"]
        )
        fig, ax = plt.subplots()
        sns_plot = sns.boxplot(data=df[variavel], ax=ax)
        sns_plot.set(ylabel=ylabels[variavel])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, width=800)
    elif graph_type == "Histograma":
        ylabels = {"IDADEMAE": "Idade da mãe", "PESO": "Peso do bebê"}
        variavel = st.selectbox(
            "Selecione a variável a ser visualizada", ["IDADEMAE", "PESO"]
        )
        fig, ax = plt.subplots()
        sns_plot = sns.histplot(
            data=df, alpha=0.25, kde=True, element="step", bins=20, x=variavel
        )
        sns_plot.set(ylabel="Frequência", xlabel=ylabels[variavel])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, width=800)


def plotagem_grafico_comparativo(df, hue_sexo=False, crosstab=False):
    if "eixo_x" not in st.session_state:
        st.session_state.eixo_x = None
    if "eixo_y" not in st.session_state:
        st.session_state.eixo_y = None

    plt.close("all")
    plt.rc("figure", figsize=(10, 8))

    col1, col2 = st.columns([1, 1])

    # Definir opções baseadas nos parâmetros
    if hue_sexo:
        opcoes_x = ["CONSULTAS", "SEXO", "ESTCIVMAE", "GESTACAO", "CAT_APGAR5"]
    else:
        opcoes_x = ["CONSULTAS", "SEXO", "ESTCIVMAE", "GESTACAO"]

    opcoes_y = ["APGAR5", "APGAR1", "IDADEMAE"]

    # Caso Crosstab
    if crosstab:
        cruzada_viz = None
        col1, col2, col3 = st.columns([1, 2, 1])
        col2.write("#### Tabela Cruzada (Crosstab)")
        cross_tab_viz = st.radio(
            "Selecione a visualizacao da tabela cruzada",
            ["Valores Absolutos", "Porcentagem"],
            horizontal=True,
        )
        if cross_tab_viz == "Valores Absolutos":
            cruzada = pd.crosstab(
                df[st.session_state.eixo_x], df[st.session_state.eixo_y]
            )
            st.dataframe(cruzada)
            return cruzada
        if cross_tab_viz == "Porcentagem":
            cruzada = pd.crosstab(
                df[st.session_state.eixo_x], df[st.session_state.eixo_y]
            )
            # Na seção de porcentagem do crosstab:
            cruzada_pct = cruzada.div(cruzada.sum(axis=1), axis=0) * 100
            st.dataframe(cruzada_pct.style.format("{:.1f}%"))
            return cruzada_pct

    if crosstab == False:
        # Selectboxes COMUNS para ambos os casos
        eixo_x = col1.selectbox(
            "Selecione a variável para o eixo X",
            opcoes_x,
            key="eixo_x_select_hue" if hue_sexo else "eixo_x_select",
        )
        st.session_state.eixo_x = eixo_x

        eixo_y = col2.selectbox(
            "Selecione a variável para o eixo Y",
            opcoes_y,
            key="eixo_y_select_hue" if hue_sexo else "eixo_y_select",
        )
        st.session_state.eixo_y = eixo_y

        # Verificar se as colunas existem
        if eixo_x not in df.columns or eixo_y not in df.columns:
            st.error("Alguma coluna selecionada não existe no DataFrame")
            return

        # Criar figura APÓS a seleção
        fig, ax = plt.subplots()

        # Caso normal ou com hue
        if hue_sexo:
            sns.barplot(
                data=df,
                x=eixo_x,
                y=eixo_y,
                hue="SEXO",
                ax=ax,
            )
            ax.set_title(f"{eixo_y} por {eixo_x} (por Sexo)")
        else:
            sns.barplot(data=df, x=eixo_x, y=eixo_y, ax=ax)
            ax.set_title(f"{eixo_y} por {eixo_x}")

        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)
        st.image(buf, width=800)


def main():
    success_placeholder = st.empty()
    with st.sidebar:
        selecao_atividade = st.selectbox(
            "Selecione a atividade desejada....",
            ["Atividade 1", "Atividade 2", "Atividade 3"],
            key="selecao_atividade_key",
        )
        st.caption(
            "Em aula foram dadas três atividades para serem realizadas utilizando esta base de dados, utilize a seleção acima para visualizar os exercícios desempenhados de acordo com a atividade selecionada"
        )
    if selecao_atividade == "Atividade 1":
        st.markdown(
            """
            <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">
            <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 2.5em;'>
                <strong>Criando um marcador de maus pagadores á partir de uma base de pagamentos</strong>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(
            "Conjunto de gráficos exibindo o `value_counts` de variáveis críticas de nossa base de dados",
            expanded=False,
        ):
            colunas_necessarias = ["APGAR1", "APGAR5", "SEXO", "ESTCIVMAE", "GESTACAO"]
            if all(coluna in df.columns for coluna in colunas_necessarias):
                col1, col2 = st.columns([1, 1])
                with col1:
                    plotagem_de_graficos_contagem(graph_type="Primário")
                with col2:
                    plotagem_de_graficos_contagem(graph_type="Secundário")

            st.subheader("Relação das variáveis categóricas com as variáveis numéricas")
            plotagem_grafico_comparativo(df)
    elif selecao_atividade == "Atividade 2":
        st.markdown(
            """
            <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">
            <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 2.5em;'>
                <strong>Distribuição das variáveis quantitativas ao longo do conjunto de dados</strong>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(
            """Visualizações em Boxplot das variáveis `"IDADEMAE"` e `"PESO"` """
        ):
            visualizacao_grafico_atividade_2 = st.radio(
                "Selecione o modelo de visualização que você deseja",
                ["Boxplot", "Histograma"],
                horizontal=True,
                key="visualizacao_atividade_2_key",
            )
            plotagem_de_graficos_contagem(visualizacao_grafico_atividade_2)
    elif selecao_atividade == "Atividade 3":
        st.markdown(
            """
            <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">
            <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 2.5em;'>
                <strong>Relação de variáveis categóricas x numéricas</strong>
                </h1>
        <p style='text-align: center; font-family: "Kantumruy Pro", sans-serif;'>
            Ponderação sobre impacto de variáveis como "GESTACAO" na categoria do APGAR5 que o infante se encontra
        </p>
            """,
            unsafe_allow_html=True,
        )
        df_work = df.copy()
        progress_bar = st.progress(0)
        status_text = st.empty()
        with st.spinner("Categorizando as medidas de APGAR5"):
            for i in range(4):
                time.sleep(1)
                progress_bar.progress((i + 1) * 25)
                status_text.text(f"Processando...{((i+1) * 25)}% completo")
            df_work.loc[
                (df_work["APGAR5"] >= 8) & (df_work["APGAR5"] <= 10),
                "CAT_APGAR5",
            ] = "normal"

            df_work.loc[
                (df_work["APGAR5"] >= 6) & (df_work["APGAR5"] <= 7),
                "CAT_APGAR5",
            ] = "asfixia leve"

            df_work.loc[
                (df_work["APGAR5"] >= 4) & (df_work["APGAR5"] <= 5),
                "CAT_APGAR5",
            ] = "asfixia moderada"

            df_work.loc[
                (df_work["APGAR5"] >= 0) & (df_work["APGAR5"] <= 3),
                "CAT_APGAR5",
            ] = "asfixia severa"

            progress_bar.empty()
            status_text.empty()

        success_placeholder.success(
            """A variável `"CAT_APGAR5"` foi criada, com o intuíto de armazenar as categorias de `"APGAR5"` """,
        )
        with st.expander("Conjunto de dados atual", expanded=False):
            st.subheader("Visualização do dataframe")
            st.dataframe(df_work)
            st.markdown("---")
            st.subheader(
                "Visualização gráfica dos níveis de asfixia dos recém nascidos por tempo de gestação"
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                plotagem_grafico_comparativo(df_work, hue_sexo=True)
            with col2:
                tabela_valores_cruzados = plotagem_grafico_comparativo(
                    df_work, hue_sexo=True, crosstab=True
                )
            st.markdown("---")
            # Cores que indicam gravidade clinicamente
            paleta_clinical = ["#2E86AB", "#F18F01", "#A23B72", "#C73E1D"]

            st.subheader(
                "Impacto do periodo de gestação com a taxa de nascidos prematuros"
            )
            plt.close("all")
            fig, ax = plt.subplots()
            if st.session_state.eixo_y == "APGAR5":
                paleta_clinical = ["#2E86AB", "#F18F01", "#A23B72", "#C73E1D"]
                sns.countplot(
                    df_work,
                    x=st.session_state.eixo_x,
                    hue="CAT_APGAR5",
                    ax=ax,
                    palette=paleta_clinical,
                )
            else:
                sns.countplot(
                    df_work,
                    x=st.session_state.eixo_x,
                    hue=st.session_state.eixo_y,
                    ax=ax,
                )
            eixo_x_gestacao = ["22 a 27", "28 a 31", "32 a 36", "37 a 41", "42+", "-22"]
            plt.xticks(rotation=45)
            plt.legend(loc="lower left", bbox_to_anchor=(-0.05, -0.2), ncol=6)
            if st.session_state.eixo_x == "GESTACAO":
                plt.gca().set_xticklabels(eixo_x_gestacao)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.image(buf, width=650)

        time.sleep(4)
        success_placeholder.empty()
        with st.expander("Analisando influência do peso do bebê em sua saúde", expanded=False):
            tab_peso_apgar = pd.crosstab(df_work["PESO"], df_work["APGAR5"])
            st.subheader("Visualizando tabela cruzada entre Peso e APGAR5")
            st.dataframe(tab_peso_apgar)
            st.markdown("---")
            
            # Sliders
            col1, col2 = st.columns(2)
            
            with col1:
                apgar_min, apgar_max = st.slider(
                    "Faixa de APGAR5:",
                    min_value=int(df_work["APGAR5"].min()),
                    max_value=int(df_work["APGAR5"].max()),
                    value=(7, 10),  # Valores normais por padrão
                    help="Escore APGAR5 (0-10)"
                )
            
            with col2:
                peso_min, peso_max = st.slider(
                    "Faixa de Peso (g):",
                    min_value=int(df_work["PESO"].min()),
                    max_value=int(df_work["PESO"].max()),
                    value=(2500, 4000),  # Peso normal por padrão
                    help="Peso ao nascer em gramas"
                )
            
            # Filtragem
            df_filtrado = df_work.loc[
                (df_work["APGAR5"] >= apgar_min) & 
                (df_work["APGAR5"] <= apgar_max) &
                (df_work["PESO"] >= peso_min) & 
                (df_work["PESO"] <= peso_max)
            ]
            
            # Visualização com múltiplos gráficos
            plt.close("all")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Histograma de Peso por APGAR5
            sns.histplot(data=df_filtrado, x="PESO", hue="APGAR5", bins=20, alpha=0.6, ax=ax1, palette="coolwarm")
            ax1.set_title("Distribuição de Peso por APGAR5")
            ax1.set_xlabel("Peso (g)")
            
            # 2. Boxplot de Peso por APGAR5
            sns.boxplot(data=df_filtrado, x="APGAR5", y="PESO", ax=ax2, palette="viridis")
            ax2.set_title("Distribuição de Peso por Escore APGAR5")
            ax2.set_ylabel("Peso (g)")
            
            # 3. Scatter plot
            sns.scatterplot(data=df_filtrado, x="PESO", y="APGAR5", alpha=0.6, ax=ax3, hue="APGAR5", palette="Spectral")
            ax3.set_title("Relação Peso vs APGAR5")
            ax3.set_xlabel("Peso (g)")
            ax3.set_ylabel("APGAR5")
            
            # 4. Gráfico de violino
            sns.violinplot(data=df_filtrado, x="APGAR5", y="PESO", ax=ax4, palette="Set2")
            ax4.set_title("Densidade de Peso por APGAR5")
            ax4.set_ylabel("Peso (g)")
            
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
            buf.seek(0)
            st.image(buf, width=800)
            
            # Estatísticas
            st.info(f"""
            **Estatísticas da faixa selecionada:**
            - 📊 **Registros:** {len(df_filtrado)} 
            - ⚖️ **Peso médio:** {df_filtrado['PESO'].mean():.0f}g
            - 🏥 **APGAR5 médio:** {df_filtrado['APGAR5'].mean():.1f}
            - 📈 **Correlação Peso-APGAR5:** {df_filtrado['PESO'].corr(df_filtrado['APGAR5']):.3f}
            """)


if __name__ == "__main__":
    main()
