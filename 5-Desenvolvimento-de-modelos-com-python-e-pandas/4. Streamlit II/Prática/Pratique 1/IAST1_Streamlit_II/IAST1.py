# imports do app

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO

# Funções que compõem o app


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


# Função para a criação de seletores de variáveis categóricas
def selecao_valores_categoricos(relatorio, col, selecionados, verificacao):
    if verificacao == True:
        return relatorio
    else:
        return relatorio[relatorio[col].isin(selecionados)]


# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(
        page_title="Telemarketing analisys",
        page_icon="Atari-Logo.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# Configurações de exbição da página de streamlit
st.set_page_config(
    page_title="Análise de dados bancários",
    page_icon="nerv.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apresenta a imagem na barra lateral da aplicação
image = Image.open("Atari-Logo.png")
st.sidebar.image(image)


# Cabeçalho do app
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Tourney&display=swap" rel="stylesheet">
    <div style="
        font-family: 'Tourney', sans-serif; 
        color: #06F906; 
        font-size: 30px; 
        text-align: center; 
        border: 2px solid #06F906; 
        padding: 10px; 
        width: fit-content; 
        margin: auto;
        margin-bottom: 20px;
        border-radius: 10px;">
        Avaliação comparativa dos dados brutos e filtrados
    </div>
    """,
    unsafe_allow_html=True,
)


# Selecionando tema escuro para as visualizações
sns.set_theme(style="dark")

# Seção para upload de um banco de dados
st.sidebar.markdown("Faça o upload de um arquivo CSV ou xlsx:")
data_file = st.sidebar.file_uploader("Dataset para análise", type=["csv", "xlsx"])

if data_file is not None:

    # Leitura do banco de dados
    dados_bancarios_raw = load_data(data_file)

    with st.sidebar.form(key="my_form"):

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

        # Criação de lista com todos os valores únicos de profissões
        # Adicionando um seletor de múltiplas profissões para o usuário
        jobs_list = dados_bancarios_raw["job"].unique().tolist()
        jobs_all = st.checkbox(
            "Selecionar todas as profissões", value=True, key="checkbox_job"
        )
        jobs_selected = st.multiselect(
            "Profissões",
            jobs_list,
            default=jobs_list if jobs_all else [],
            key="multiselect_job",
        )

        st.markdown("---")

        # Criação de lista com todos os valores únicos de estado civil
        # Adicionando um seletor de múltiplas estados civis
        marital_list = dados_bancarios_raw["marital"].unique().tolist()
        marital_all = st.checkbox(
            "Selecionar todos os estados civis", value=True, key="checkbox_marital"
        )
        marital_selected = st.multiselect(
            "Estado civil",
            marital_list,
            default=marital_list if marital_all else [],
            key="multiselect_marital",
        )

        st.markdown("---")

        # Criação de lista com todos os valores únicos de educação
        # Adicionando um seletor de múltiplas níveis de educação
        education_list = dados_bancarios_raw["education"].unique().tolist()
        education_all = st.checkbox(
            "Selecionar todos os níveis de educação",
            value=True,
            key="checkbox_education",
        )
        education_selected = st.multiselect(
            "Educação",
            education_list,
            default=education_list if education_all else [],
            key="multiselect_education",
        )

        st.markdown("---")

        # Criação de lista com todos os valores únicos de status de crédito (DEFAULT)
        # Adicionando um seletor de múltiplos status de crédito
        default_list = dados_bancarios_raw["default"].unique().tolist()
        default_all = st.checkbox(
            "Selecionar todos os status de crédito", value=True, key="checkbox_default"
        )
        default_selected = st.multiselect(
            "Status de crédito",
            default_list,
            default=default_list if default_all else [],
            key="multiselect_default",
        )

        st.markdown("---")

        # Criação de lista com todos os valores únicos de status de financiamento imobiliario (HOUSING)
        # Adicionando um seletor de múltiplos status de habitação
        housing_list = dados_bancarios_raw["housing"].unique().tolist()
        housing_all = st.checkbox(
            "Selecionar todos os status de financiamento de imóvel",
            value=True,
            key="checkbox_housing",
        )
        housing_selected = st.multiselect(
            "Status de habitação",
            housing_list,
            default=housing_list if housing_all else [],
            key="multiselect_housing",
        )

        st.markdown("---")

        # Criação de lista com todos os valores únicos de status de crédito (LOAN)
        # Adicionando um seletor de múltiplos status de crédito
        loan_list = dados_bancarios_raw["loan"].unique().tolist()
        loan_all = st.checkbox(
            "Selecionar todos os status de crédito", value=True, key="checkbox_loan"
        )
        loan_selected = st.multiselect(
            "Status de crédito",
            loan_list,
            default=loan_list if loan_all else [],
            key="multiselect_loan",
        )

        st.markdown("---")

        # Criação de lista com todos os valores únicos de contato (CONTACT)
        # Adicionando um seletor de múltiplos contatos
        contact_list = dados_bancarios_raw["contact"].unique().tolist()
        contact_all = st.checkbox(
            "Selecionar todos os contatos", value=True, key="checkbox_contact"
        )
        contact_selected = st.multiselect(
            "Contato",
            contact_list,
            default=contact_list if contact_all else [],
            key="multiselect_contact",
        )

        st.markdown("---")

        # Criação de lista com todos os valores únicos de mês de contato (MONTH)
        # Adicionando um seletor de múltiplos meses de contato
        month_list = dados_bancarios_raw["month"].unique().tolist()
        month_all = st.checkbox(
            "Selecionar todos os meses de contato", value=True, key="checkbox_month"
        )
        month_selected = st.multiselect(
            "Mês de contato",
            month_list,
            default=month_list if month_all else [],
            key="multiselect_month",
        )

        st.markdown("---")

        # Criação de lista com todos os valores únicos de dia de contato (DAY_OF_WEEK)
        # Adicionando um seletor de múltiplos dias de contato
        day_of_week_list = dados_bancarios_raw["day_of_week"].unique().tolist()
        days_of_week_all = st.checkbox(
            "Selecionar todos os dias de contato",
            value=True,
            key="checkbox_day_of_week",
        )
        day_of_week_selected = st.multiselect(
            "Dia de contato",
            day_of_week_list,
            default=day_of_week_list if days_of_week_all else [],
            key="multiselect_day_of_week",
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
            day_of_week_selected,
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

    # Definição de proporção percentual da variável target (y) para os dados brutos
    dados_bancarios_raw_percentual_target = (
        dados_bancarios_raw.y.value_counts(normalize=True).to_frame() * 100
    )
    dados_bancarios_raw_percentual_target = (
        dados_bancarios_raw_percentual_target.sort_index()
    )

    # Definição de proporção percentual da variável target (y) para os dados filtrados
    try:
        dados_bancarios_filtrados_percentual_target = (
            dados_bancarios_filtrados.y.value_counts(normalize=True).to_frame() * 100
        )
        dados_bancarios_filtrados_percentual_target = (
            dados_bancarios_filtrados_percentual_target.sort_index()
        )
    except:
        st.error("Erro no filtro")

    # Criação de duas colunas para comportar as informações de proporção dos dataframes
    col1, col2 = st.columns(2)

    # Exibição da proporção percentual da variável target (y) para os dados brutos
    col1.write("### Proporção original")
    col1.write(dados_bancarios_raw_percentual_target)

    # Exibição da proporção percentual da variável target (y) para os dados filtrados
    col2.write("### Proporção filtrada")
    col2.write(dados_bancarios_filtrados_percentual_target)

    # Criação de gráfico para comparação entre os dados brutos e filtrados
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    # Definindo o gráfico de acordo com a escolha do usuário
    if graph_type == "Barras":

        # Gráfico de barras para os dados brutos
        sns.barplot(
            x=dados_bancarios_raw_percentual_target.index,
            y="proportion",
            data=dados_bancarios_raw_percentual_target,
            palette="rocket",
            ax=ax[0],
        )
        # Configuração da visualização do gráfico de barras
        ax[0].set_ylabel("Proporção (%)")
        ax[0].set_xlabel("Dados Brutos")
        ax[0].bar_label(ax[0].containers[0])
        ax[0].set_ylim(0, 100)
        ax[0].set_title("Dados Brutos", fontweight="bold")
        fig.tight_layout()

        # Gráfico de barras para os dados filtrados
        sns.barplot(
            x=dados_bancarios_filtrados_percentual_target.index,
            y="proportion",
            data=dados_bancarios_filtrados_percentual_target,
            palette="rocket",
            ax=ax[1],
        )

        # Configuração da visualização do gráfico de barras
        ax[1].set_ylabel("Proporção (%)")
        ax[1].bar_label(ax[1].containers[1])
        ax[1].set_ylim(0, 100)
        ax[1].set_xlabel("Dados Filtraodos")
        ax[1].set_title("Dados Filtrados", fontweight="bold")
        fig.tight_layout()  # Ajusta automaticamente os subplots para evitar sobreposição

        # Exibição do gráfico de barras
        st.pyplot(plt)

    # Se o usuário escolher gráfico de pizza, o gráfico será gerado com os dados filtrados
    else:
        # Gráfico de pizza para os dados brutos
        dados_bancarios_raw_percentual_target.plot.pie(
            y="proportion",
            ax=ax[0],
            autopct="%1.1f%%",
            startangle=90,
            legend=False,
            title="Dados brutos",
        )

        # Configuração da visualização do gráfico de pizza
        ax[0].set_ylabel("Proporção (%)")
        ax[0].set_xlabel("Dados brutos")
        ax[0].set_title("Dados brutos", fontweight="bold")
        fig.tight_layout()

        # Gráfico de pizza para os dados filtrados
        dados_bancarios_filtrados_percentual_target.plot.pie(
            y="proportion",
            ax=ax[1],
            autopct="%1.1f%%",
            startangle=90,
            legend=False,
            title="Dados filtrados",
        )

        # Configuração da visualização do gráfico de pizza
        ax[1].set_ylabel("Proporção (%)")
        ax[1].set_xlabel("Dados filtrados")
        ax[1].set_title("Dados filtrados", fontweight="bold")
        fig.tight_layout()

        # Exibição do gráfico de pizza
        st.pyplot(plt)
