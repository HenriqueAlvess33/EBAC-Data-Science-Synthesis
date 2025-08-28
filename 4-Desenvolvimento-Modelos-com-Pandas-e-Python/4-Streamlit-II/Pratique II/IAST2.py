import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO

# Funções que compõem o app


def visualizacao_distribuicao_variavel_y(
    _df_inteiro, _df_filtrado, tipo_de_grafico, variavel_categorica
):
    # Cálculo das proporções
    _df_inteiro_proporcao = (
        _df_inteiro[variavel_categorica]
        .value_counts(normalize=True)
        .mul(100)
        .reset_index()
    )
    _df_inteiro_proporcao.columns = ["Categoria", "Proporção"]

    _df_filtrado_proporcao = (
        _df_filtrado[variavel_categorica]
        .value_counts(normalize=True)
        .mul(100)
        .reset_index()
    )
    _df_filtrado_proporcao.columns = ["Categoria", "Proporção"]

    # Layout de colunas para tabelas
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Proporção Original")
        st.dataframe(_df_inteiro_proporcao, hide_index=True)

    with col2:
        st.write("### Proporção Filtrada")
        st.dataframe(_df_filtrado_proporcao, hide_index=True)

    # Visualização comparativa
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    if tipo_de_grafico == "Barras":
        # Gráfico 1 - Dados brutos
        bar1 = sns.barplot(
            data=_df_inteiro_proporcao,
            x="Categoria",
            y="Proporção",
            palette="viridis",
            ax=ax[0],
        )
        ax[0].set_title("Distribuição Original\n", fontweight="bold")
        ax[0].set_ylim(0, 100)

        # Adicionar labels nas barras (forma segura)
        for container in bar1.containers:
            bar1.bar_label(container, fmt="%.1f%%", padding=3)

        # Gráfico 2 - Dados filtrados
        bar2 = sns.barplot(
            data=_df_filtrado_proporcao,
            x="Categoria",
            y="Proporção",
            palette="viridis",
            ax=ax[1],
        )
        ax[1].set_title("Distribuição Filtrada\n", fontweight="bold")
        ax[1].set_ylim(0, 100)

        for container in bar2.containers:
            bar2.bar_label(container, fmt="%.1f%%", padding=3)

    elif tipo_de_grafico == "Pizza":
        # Implementação alternativa para pizza
        ax[0].pie(
            _df_inteiro_proporcao["Proporção"],
            labels=_df_inteiro_proporcao["Categoria"],
            autopct="%1.1f%%",
        )
        ax[0].set_title("Distribuição Original")

        ax[1].pie(
            _df_filtrado_proporcao["Proporção"],
            labels=_df_filtrado_proporcao["Categoria"],
            autopct="%1.1f%%",
        )
        ax[1].set_title("Distribuição Filtrada")

    plt.tight_layout()

    # Exibição no Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_container_width=True)

    # Adicionar métrica de desvio
    st.metric(
        "Desvio Máximo na Proporção",
        f"{abs(_df_inteiro_proporcao['Proporção'].iloc[0] - _df_filtrado_proporcao['Proporção'].iloc[0]):.2f}%",
    )


# Função para converter o df em uma planila excel
def df_to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Sheet1")
    writer.close()
    processed_data = output.getvalue()
    return processed_data


# Função para ler os dados
@st.cache_data()
def load_data(file_data):
    return pd.read_csv(file_data, sep=";")


def selecao_valores_categoricos(relatorio, col, selecionados, select_all):
    """
    Filtra dataframe baseado na seleção do usuário

    Args:
        relatorio: DataFrame para filtrar
        col: Coluna categórica para filtrar
        selecionados: Lista de valores selecionados
        select_all: Se True, retorna cópia do dataframe (sem filtrar esta coluna)

    Returns:
        NOVO DataFrame filtrado (cópia)
    """
    try:
        # SEMPRE trabalhar com cópia para não modificar o original
        _df_trabalho = relatorio.copy()

        if select_all:
            # "Selecionar todos" marcado - retorna cópia completa (sem filtrar esta coluna)
            return _df_trabalho

        elif selecionados:  # Verifica se há valores selecionados
            # Filtra apenas pelos valores selecionados
            return _df_trabalho[_df_trabalho[col].isin(selecionados)]

        else:
            # Nada selecionado e "Selecionar todos" desmarcado
            st.warning(
                f"⚠️ Nenhum valor selecionado para {col}. Retornando dataframe vazio."
            )
            return _df_trabalho.iloc[0:0]  # DataFrame vazio com mesma estrutura

    except KeyError:
        st.error(f"Coluna '{col}' não encontrada no dataframe.")
        return _df_trabalho
    except Exception as e:
        st.error(f"Erro inesperado ao filtrar {col}: {str(e)}")
        return _df_trabalho


def filtragem_de_dados(
    dataframe, variavel, string_para_labels, placeholder=None, key_suffix=""
):
    valores_unicos = sorted(dataframe[variavel].dropna().unique().tolist())

    # Checkbox
    select_all = st.checkbox(
        f"📋 Selecionar TODAS as {string_para_labels.lower()}",
        value=True,
        key=f"select_all_{variavel}_{key_suffix}",
    )

    # Lógica condicional para o multiselect
    if select_all:
        # Quando select_all=True, mostra multiselect desabilitado (melhor UX)
        chosen_variables = st.multiselect(
            f"✅ {string_para_labels} (todas selecionadas)",
            valores_unicos,
            default=valores_unicos,
            key=f"multiselect_{variavel}_{key_suffix}",
            placeholder=placeholder or f"Selecione {string_para_labels.lower()}",
        )
        return valores_unicos, True
    else:
        # Quando select_all=False, mostra multiselect habilitado
        chosen_variables = st.multiselect(
            f"🔍 {string_para_labels}",
            valores_unicos,
            default=valores_unicos,
            disabled=False,
            key=f"multiselect_{variavel}_{key_suffix}",
            placeholder=placeholder or f"Selecione {string_para_labels.lower()}",
        )
        return chosen_variables, False


# Função principal da aplicação
def main():
    # Configura o título da aplicação
    st.set_page_config(
        page_title="Ánalise descritiva",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="varig_icon.png",
    )
    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">

    <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 3em;'>
        <strong>Análise descritiva dos dados de um dataset bancário</strong>
    </h1>
    """,
        unsafe_allow_html=True,
    )


main()

with st.sidebar:
    # Apresenta a imagem na barra lateral da aplicação
    image = Image.open("Atari-Logo.png")
    st.image(image)

    # Seção para upload de um banco de dados
    st.markdown("Faça o upload de um arquivo CSV ou xlsx:")
    data_file = st.file_uploader("Dataset para análise", type=["csv", "xlsx"])

if data_file is not None:

    # Leitura do banco de dados
    dados_bancarios_raw = load_data(data_file)
    filtros = [
        ("job", "Profissões"),
        ("marital", "Estado civil"),
        ("education", "Nivel de escolaridade"),
        ("default", "default"),
        ("housing", "Status de habitação"),
        ("loan", "Status de crédito"),
        ("contact", "Contato"),
        ("month", "Mês de contato"),
        ("day_of_week", "Dia de contatp"),
    ]

    with st.sidebar:
        with st.form(key="my_form"):
            # Definindo os valores mínimos e máximos para as idades
            min_age = int(dados_bancarios_raw["age"].min())
            max_age = int(dados_bancarios_raw["age"].max())

            # Formatando o seletor de idades para filtragem e criação de gráficos
            idades = st.slider(
                label="Idades",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age),
                step=1,
                format="%d",
            )
            st.markdown("---")

            # Adicionando um seletor de múltiplas profissões
            jobs_selected, jobs_all = filtragem_de_dados(
                variavel="job",
                string_para_labels="Profissões",
                dataframe=dados_bancarios_raw,
            )

            st.markdown("---")

            # Adicionando um seletor de múltiplas estados civis
            marital_selected, marital_all = filtragem_de_dados(
                variavel="marital",
                string_para_labels="Estado civil",
                dataframe=dados_bancarios_raw,
            )

            st.markdown("---")

            # Adicionando um seletor de múltiplas estados civis
            education_selected, education_all = filtragem_de_dados(
                variavel="education",
                string_para_labels="Nivel de escolaridade",
                dataframe=dados_bancarios_raw,
            )

            st.markdown("---")

            # Adicionando um seletor de múltiplas estados civis
            default_selected, default_all = filtragem_de_dados(
                variavel="default",
                string_para_labels="Default",
                dataframe=dados_bancarios_raw,
            )

            st.markdown("---")

            # Adicionando um seletor de múltiplas estados civis
            housing_selected, housing_all = filtragem_de_dados(
                variavel="housing",
                string_para_labels="Status de habitação",
                dataframe=dados_bancarios_raw,
            )

            st.markdown("---")

            # Adicionando um seletor de múltiplas estados civis
            loan_selected, loan_all = filtragem_de_dados(
                variavel="loan",
                string_para_labels="Status de crédito",
                dataframe=dados_bancarios_raw,
            )

            st.markdown("---")

            # Adicionando um seletor de múltiplas estados civis
            contact_selected, contact_all = filtragem_de_dados(
                variavel="contact",
                string_para_labels="Contato",
                dataframe=dados_bancarios_raw,
            )

            st.markdown("---")

            # Adicionando um seletor de múltiplas estados civis
            month_selected, month_all = filtragem_de_dados(
                variavel="month",
                string_para_labels="Mês de contato",
                dataframe=dados_bancarios_raw,
            )

            st.markdown("---")

            # Adicionando um seletor de múltiplas estados civis
            days_of_week_selected, days_of_week_all = filtragem_de_dados(
                variavel="day_of_week",
                string_para_labels="Dia de contato",
                dataframe=dados_bancarios_raw,
            )

            st.markdown("---")

            graph_type = st.radio(
                "Tipo de gráfico", ("Barras", "Pizza"), index=0, horizontal=True
            )

            # Botão de aplicar filtro
            submit_button = st.form_submit_button(label="Filtrar dados")

    dados_bancarios_filtrados = (
        dados_bancarios_raw.query("age>=@idades[0] and age <= @idades[1]")
        .pipe(selecao_valores_categoricos, "job", jobs_selected, jobs_all)
        .pipe(selecao_valores_categoricos, "marital", marital_selected, marital_all)
        .pipe(
            selecao_valores_categoricos, "education", education_selected, education_all
        )
        .pipe(selecao_valores_categoricos, "default", default_selected, default_all)
        .pipe(selecao_valores_categoricos, "housing", housing_selected, housing_all)
        .pipe(selecao_valores_categoricos, "loan", loan_selected, loan_all)
        .pipe(selecao_valores_categoricos, "contact", contact_selected, contact_all)
        .pipe(selecao_valores_categoricos, "month", month_selected, month_all)
        .pipe(
            selecao_valores_categoricos,
            "day_of_week",
            days_of_week_selected,
            days_of_week_all,
        )
    )

    # Atribuindo a conversão do dataframe filtrado para o formato excel a uma variável
    df_xlsx = df_to_excel(dados_bancarios_filtrados)

    # Botão de download do dataframe filtrado em excel
    st.download_button(
        data=df_xlsx,
        label="🟢⬇️ Faça o download do Dataframe filtrado em excel",
        file_name="dados filtrados.xlsx",
    )

    # Guardando informação do tamanho do dataset
    dimensao_dataset_raw = dados_bancarios_raw.shape

    # Guardando informação do tamanho do dataset filtrado
    dimensao_dataset_filtrado = dados_bancarios_filtrados.shape

    # Declarando para o usuário do que se trata o dataset
    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Orbitron&display=swap" rel="stylesheet">
    <div style="
        font-family: 'Orbitron', sans-serif; 
        color: #FFFFFF; 
        font-size: 24px; /* Tamanho equivalente a ## do Markdown */
        text-align: right; 
        border: 2px solid #FFFFFF; 
        padding: 20px; 
        width: fit-content; 
        margin-left: auto; 
        margin-right: 0;
        margin-top: 20px;
        margin-bottom: 20px;
        border-radius: 10px;">
        Dados Bancários sem alterações
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Exibindo o dataframe original e escrevendo suas dimensões
    st.markdown(
        f"**Tamanho do dataset:** {dimensao_dataset_raw[0]} linhas e {dimensao_dataset_raw[1]} colunas"
    )
    st.dataframe(
        dados_bancarios_raw.head(n=5).style.set_properties(
            **{"background-color": "#0a0f25", "color": "#f8f9fa"}
        )
    )

    # Exibindo o dataframe original e escrevendo suas dimensões
    st.markdown(
        f"**Tamanho do dataset:** {dimensao_dataset_raw[0]} linhas e {dimensao_dataset_raw[1]} colunas"
    )

    # Declarando para o usuário do que se trata o dataset
    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Orbitron&display=swap" rel="stylesheet">
    <div style="
        font-family: 'Orbitron', sans-serif; 
        color: #FFFFFF; 
        font-size: 24px; /* Tamanho equivalente a ## do Markdown */
        text-align: right; 
        border: 2px solid #FFFFFF; 
        padding: 20px; 
        width: fit-content; 
        margin-left: auto; 
        margin-right: 0;
        margin-top: 20px;
        margin-bottom: 20px;
        border-radius: 10px;">
        Dados Bancários Filtrados
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Escrevendo as dimensões do dataframe filtrado
    st.markdown(
        f"**Tamanho do dataset:** {dimensao_dataset_filtrado[0]} linhas e {dimensao_dataset_filtrado[1]} colunas"
    )

    # Exibindo o dataframe filtrado
    st.dataframe(
        dados_bancarios_filtrados.head(n=5).style.set_properties(
            **{"background-color": "#0a0f25", "color": "#f8f9fa"}
        )
    )

    with st.expander(
        "Comparação entre proporções de dataframe filtrado e inteiro", expanded=False
    ):
        visualizacao_distribuicao_variavel_y(
            _df_inteiro=dados_bancarios_raw,
            _df_filtrado=dados_bancarios_filtrados,
            tipo_de_grafico=graph_type,
            variavel_categorica="y",
        )
