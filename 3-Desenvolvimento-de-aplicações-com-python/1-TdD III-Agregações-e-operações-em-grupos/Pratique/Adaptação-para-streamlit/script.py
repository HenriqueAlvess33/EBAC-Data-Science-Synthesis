import streamlit as st
import asyncio
import pandas as pd
import numpy as np
from datetime import date
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import requests
from functools import lru_cache

# ------------------------------------------------------
# CONFIGURAÇÃO INICIAL DA PÁGINA STREAMLIT
# ------------------------------------------------------
st.set_page_config(
    page_title="Preparando a base",  # Título exibido na aba do navegador
    layout="wide",  # Layout expandido (usa largura total da tela)
    initial_sidebar_state="expanded",  # Abre a barra lateral automaticamente
    page_icon="varig_icon.png",  # Ícone da aba
)

# ------------------------------------------------------
# DEFININDO VARIÁVEIS DE SESSÃO
# ------------------------------------------------------
# Usadas para armazenar os valores selecionados pelos usuários e evitar reset
session_defaults = {
    "corte_longitude_aplicar": None,
    "corte_area_aplicar": None,
    "corte_mun_residencia_aplicar": None,
}

# Inicializa no session_state se ainda não existir
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------


def visualizacao_var_numerica_por_faixa(
    variavel, session_state, nome_coluna, dataframe
):
    """
    Cria faixas (bins) com base em um valor de corte selecionado pelo usuário,
    divide os dados em 'acima' e 'abaixo' do corte, e exibe estatísticas
    descritivas agrupadas por essas faixas.
    """
    # Slider para escolher ponto de corte
    corte = st.slider(
        "Selecione aonde iremos diviidir as faixas",
        help="Duas faixas serão criadas: uma abaixo e outra acima do valor escolhido",
        min_value=dataframe[variavel].min(),
        max_value=dataframe[variavel].max(),
        key=f"slider_key{nome_coluna}",
    )

    # Botão para aplicar o corte
    if st.button(
        "Aplicar valor selecionado", key=f"botao_para_confirmar_key_{nome_coluna}"
    ):
        session_state = corte  # Salva no estado da sessão

    if session_state is not None:
        # Aplica o corte criando faixas com pd.cut()
        dataframe_cut = dataframe
        dataframe_cut[f"{nome_coluna}"] = pd.cut(
            dataframe["munResLat"],
            bins=[-float("inf"), session_state, float("inf")],
            labels=[f"Abaixo de {session_state}", f"Acima de {session_state}"],
        )

        # Seleciona apenas colunas numéricas
        colunas_numericas = dataframe.select_dtypes(
            include=["int64", "float64"]
        ).columns.to_list()

        # Selectbox para escolher variáveis para análise
        col1, col2 = st.columns(2)
        with col1:
            selecao_var_1 = st.selectbox(
                "Selecione a variável desejada para o agrupamento",
                index=colunas_numericas.index("IDADEMAE"),
                options=colunas_numericas,
                key=f"selecao_var_1__key_{nome_coluna}",
            )

        with col2:
            selecao_var_2 = st.selectbox(
                "Selecione a variável desejada para o agrupamento",
                index=colunas_numericas.index("QTDFILVIVO"),
                options=colunas_numericas,
                key=f"selecao_var_2_key_{nome_coluna}",
            )

        # Gera tabelas de estatísticas descritivas (soma, média, etc.)
        agrupamento_1 = dataframe_cut.groupby(nome_coluna).agg(
            {
                selecao_var_1: [
                    ("soma", "sum"),
                    ("média", "mean"),
                    ("mínimo", "min"),
                    ("máximo", "max"),
                    ("mediana", "median"),
                    ("desvio padrão", "std"),
                    ("variância", "var"),
                ],
            },
        )
        agrupamento_2 = dataframe_cut.groupby(nome_coluna).agg(
            {
                selecao_var_2: [
                    ("soma", "sum"),
                    ("média", "mean"),
                    ("mínimo", "min"),
                    ("máximo", "max"),
                    ("mediana", "median"),
                    ("desvio padrão", "std"),
                    ("variância", "var"),
                ],
            },
        )

        # Exibe os resultados em duas colunas
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(agrupamento_1)
        with col2:
            st.dataframe(agrupamento_2)


@st.cache_data
def load_data(conjunto_de_dados):
    """
    Carrega um arquivo CSV enviado pelo usuário.
    - Usa cache (@st.cache_data) para não recarregar a cada interação.
    - Retorna DataFrame pandas.
    """
    try:
        return pd.read_csv(conjunto_de_dados, sep=",")
    except Exception as e:
        st.error(f"Não foi possível carregar o arquivo selecionado. {e}")


async def temporary_success(message, duration=1):
    """
    Exibe mensagem de sucesso temporária por alguns segundos.
    """
    success_placeholder = st.empty()
    success_placeholder.success(message)
    await asyncio.sleep(duration)
    success_placeholder.empty()


# ------------------------------------------------------
# APLICAÇÃO PRINCIPAL
# ------------------------------------------------------


def main():
    # Barra lateral com seleção de atividade
    with st.sidebar:
        selecao_de_atividades = st.radio(
            "Selecione a atividade que deseja visualizar:",
            ["Atividade 1", "Atividade 2"],
            horizontal=True,
        )

    # Upload de arquivo
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Dataset para segregar clientes", type=["csv", "xlsx"]
        )
        df_original = load_data(uploaded_file)

    # ------------------------------------------------------
    # ATIVIDADE 1: VISUALIZAÇÃO DE DADOS
    # ------------------------------------------------------
    if selecao_de_atividades == "Atividade 1":
        # Título estilizado em HTML
        st.markdown(
            """
            <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">
            <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 2.5em;'>
                <strong> Visualização de um dataframe de nascidos por município </strong>
            </h1>
            """,
            unsafe_allow_html=True,
        )

        if df_original is not None:
            # Mensagem de sucesso
            asyncio.run(temporary_success("Dataframe carregado com sucesso!"))

            # Remove registros indesejados
            df_original = df_original.drop(
                df_original[
                    df_original["munResNome"] == "Município ignorado - RO"
                ].index
            )

            # Converte datas
            df_original["DTNASC"] = pd.to_datetime(
                df_original["DTNASC"], errors="coerce"
            )

            # Expander: mostra dataframe completo
            with st.expander("Visualização do Conjunto de dados"):
                st.write("#### Dados completos")
                st.dataframe(df_original)
                st.divider()

                # Tabs: Idade absoluta e idade média
                st.write("### Visualização de números absolutos")
                aba1_numeros_absolutos, aba2_media_de_idades = st.tabs(
                    ["Idade em valores absolutos", "Idade média das populações"]
                )

                # --------------------------------------
                # Primeira aba: números absolutos
                # --------------------------------------
                with aba1_numeros_absolutos:
                    col1, col2 = st.columns(2)
                    with col1:
                        # Lista de municípios
                        cidades_do_dataframe = (
                            df_original["munResNome"].value_counts().index.to_list()
                        )
                        # Checkbox: selecionar todas as cidades
                        if st.checkbox(
                            "Selecionar todas as cidades:", key="checkbox_aba1_key"
                        ):
                            filtragem_de_municipios = st.multiselect(
                                "Selecione as cidades que deseja ver",
                                default=cidades_do_dataframe,
                                options=cidades_do_dataframe,
                                key="filtragem_de_municipios_aba1_key",
                            )
                        else:
                            filtragem_de_municipios = st.multiselect(
                                "Selecionar as cidades que deseja ver",
                                cidades_do_dataframe,
                            )

                    with col2:
                        # Aplica filtro de municípios
                        if filtragem_de_municipios:
                            df_filtrado = df_original[
                                df_original["munResNome"].isin(filtragem_de_municipios)
                            ]
                        else:
                            df_filtrado = None

                        # Exibe município e idade da mãe
                        if df_filtrado is not None:
                            st.dataframe(
                                df_filtrado[["munResNome", "IDADEMAE"]].reset_index(
                                    drop=True
                                )
                            )

                # --------------------------------------
                # Segunda aba: idade média
                # --------------------------------------
                with aba2_media_de_idades:
                    col1, col2 = st.columns(2)
                    with col1:
                        cidades_do_dataframe = (
                            df_original["munResNome"].value_counts().index.to_list()
                        )
                        if st.checkbox(
                            "Selecionar todas as cidades:", key="checkbox_aba2_key"
                        ):
                            filtragem_de_municipios = st.multiselect(
                                "Selecione as cidades que deseja ver",
                                default=cidades_do_dataframe,
                                options=cidades_do_dataframe,
                            )
                        else:
                            filtragem_de_municipios = st.multiselect(
                                "Selecionar as cidades que deseja ver",
                                cidades_do_dataframe,
                                key="filtragem_de_municipios_aba2_key",
                            )

                    with col2:
                        if filtragem_de_municipios:
                            df_filtrado = df_original[
                                df_original["munResNome"].isin(filtragem_de_municipios)
                            ]
                        else:
                            df_filtrado = None

                        if df_filtrado is not None:
                            # Média da idade da mãe e do pai
                            media_idademae_idadepai = (
                                df_filtrado.groupby("munResNome")[
                                    ["IDADEMAE", "IDADEPAI"]
                                ]
                                .mean()
                                .round(2)
                            )
                            st.dataframe(media_idademae_idadepai)
                            st.divider()

                            # Gráfico comparativo
                            df_plot = media_idademae_idadepai.reset_index()
                            fig = px.bar(
                                df_plot,
                                x="munResNome",
                                y=["IDADEMAE", "IDADEPAI"],
                                barmode="group",
                                title="Idade Média por Município",
                            )
                            st.plotly_chart(fig)

                # ---------------------------------------------------------
                # VISUALIZAÇÃO EXTRA: peso médio por escolaridade e sexo
                # ---------------------------------------------------------
                st.divider()
                st.write("#### Peso médio dos bebês e dos pais por município")

                # Date input para filtrar nascimentos
                data_escolhida = st.date_input(
                    "Selecione a data",
                    value=date(2019, 6, 28),
                    min_value=date(2019, 1, 1),
                    max_value=date(2019, 12, 31),
                    format="DD/MM/YYYY",
                )

                agrupamento_idade_peso = (
                    df_original.loc[df_original["DTNASC"].dt.date == data_escolhida]
                    .groupby(["ESCMAE", "SEXO"])["PESO"]
                    .mean()
                    .reset_index()
                )

                st.dataframe(agrupamento_idade_peso)

                # Gráfico peso dos bebês
                fig_2 = px.bar(
                    agrupamento_idade_peso,
                    x="ESCMAE",
                    y=["PESO"],
                    color="SEXO",
                    barmode="group",
                    title="Peso dos bebês de acordo com a escolaridade das mães",
                )
                st.plotly_chart(fig_2)

                # Perguntas e respostas: municípios com mais/menos nascidos
                st.divider()
                st.write("#### Perguntas e Respostas")

                dicionario_meses = {
                    1: "Janeiro",
                    2: "Fevereiro",
                    3: "Março",
                    4: "Abril",
                    5: "Maio",
                    6: "Junho",
                    7: "julho",
                    8: "Agosto",
                    9: "Setembro",
                    10: "Outubro",
                    11: "Novembro",
                    12: "Dezembro",
                    0: "Todos",
                }

                mes_selecionado = st.selectbox(
                    "Selecione o mês:",
                    options=list(dicionario_meses.keys()),
                    format_func=lambda x: dicionario_meses[x],
                )

                volume_de_nascidos = st.radio(
                    "Selecione o tipo de informação que deseja acessar:",
                    ["Maior número de nascidos", "Menor número de nascidos"],
                    horizontal=True,
                )

                st.write("Qual o município com o menor número de nascimentos em 2019 ?")

                col1, col2 = st.columns(2)
                with col1:
                    # Filtra por mês, se necessário
                    if volume_de_nascidos == "Menor número de nascidos":
                        if mes_selecionado != 0:
                            df_exibicao_numero_de_nascidos = df_original.loc[
                                df_original["DTNASC"].dt.month == mes_selecionado
                            ]
                        else:
                            df_exibicao_numero_de_nascidos = df_original
                        contagem_de_nascimentos = (
                            df_exibicao_numero_de_nascidos["munResNome"]
                            .value_counts(ascending=True)
                            .head(5)
                        )
                    else:
                        if mes_selecionado != 0:
                            df_exibicao_numero_de_nascidos = df_original.loc[
                                df_original["DTNASC"].dt.month == mes_selecionado
                            ]
                        else:
                            df_exibicao_numero_de_nascidos = df_original
                        contagem_de_nascimentos = (
                            df_exibicao_numero_de_nascidos["munResNome"]
                            .value_counts(ascending=False)
                            .head(5)
                        )
                    st.table(contagem_de_nascimentos)

                with col2:
                    st.dataframe(
                        df_exibicao_numero_de_nascidos[
                            df_exibicao_numero_de_nascidos["munResNome"]
                            == contagem_de_nascimentos.index[0]
                        ]
                        .groupby("munResNome")[["IDADEMAE", "IDADEPAI"]]
                        .agg(
                            [("Mínima", "min"), ("Média", np.median), ("Máxima", "max")]
                        )
                    )

    # ------------------------------------------------------
    # ATIVIDADE 2: CRIAÇÃO DE FAIXAS E AGRUPAMENTOS
    # ------------------------------------------------------
    if selecao_de_atividades == "Atividade 2":
        st.markdown(
            """
            <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">
            <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 2.5em;'>
                <strong> Criação de faixas de valores e agrupamento de informações </strong>
            </h1>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Conjunto de dados- Análise de Natalidade - RO 2019"):

            # Cria 3 abas com diferentes critérios de corte
            aba1, aba2, aba3 = st.tabs(
                [
                    "Faixas e agrupamentos de acordo com a longitude dos municipios",
                    "Criando faixas de acordo com a área do município",
                    "Faixas de agrupamento para a variável que identifica o município de residência",
                ]
            )

            with aba1:
                slider_long = visualizacao_var_numerica_por_faixa(
                    dataframe=df_original,
                    nome_coluna="faixa_longitude",
                    session_state=st.session_state.corte_longitude_aplicar,
                    variavel="munResLat",
                )
            with aba2:
                slider_area = visualizacao_var_numerica_por_faixa(
                    dataframe=df_original,
                    nome_coluna="faixa_area",
                    session_state=st.session_state.corte_area_aplicar,
                    variavel="munResArea",
                )
            with aba3:
                slider_residencia_municipio = visualizacao_var_numerica_por_faixa(
                    dataframe=df_original,
                    nome_coluna="faixa_mun_residencia",
                    session_state=st.session_state.corte_mun_residencia_aplicar,
                    variavel="munResAlt",
                )
        with st.expander(
            """Visualização da performance da variável "idade mãe" ao longo do ano de 2019, sendo avaliada a performance atrvés das regiões imediatas """,
            expanded=True,
        ):
            imediatas = {
                "Candeias do Jamari": "Porto Velho",
                "Guajará-Mirim": "Porto Velho",
                "Itapuã do Oeste": "Porto Velho",
                "Nova Mamoré": "Porto Velho",
                "Porto Velho": "Porto Velho",
                "Ariquemes": "Ariquemes",
                "Alto Paraíso": "Ariquemes",
                "Buritis": "Ariquemes",
                "Cacaulândia": "Ariquemes",
                "Campo Novo de Rondônia": "Ariquemes",
                "Cujubim": "Ariquemes",
                "Monte Negro": "Ariquemes",
                "Rio Crespo": "Ariquemes",
                "Jaru": "Jaru",
                "Governador Jorge Teixeira": "Jaru",
                "Machadinho D'Oeste": "Jaru",
                "Theobroma": "Jaru",
                "Vale do Anari": "Jaru",
                "Alvorada D'Oeste": "Ji-Paraná",
                "Costa Marques": "Ji-Paraná",
                "Ji-Paraná": "Ji-Paraná",
                "Mirante da Serra": "Ji-Paraná",
                "Nova União": "Ji-Paraná",
                "Ouro Preto do Oeste": "Ji-Paraná",
                "Presidente Médici": "Ji-Paraná",
                "São Francisco do Guaporé": "Ji-Paraná",
                "São Miguel do Guaporé": "Ji-Paraná",
                "Seringueiras": "Ji-Paraná",
                "Teixeirópolis": "Ji-Paraná",
                "Urupá": "Ji-Paraná",
                "Vale do Paraíso": "Ji-Paraná",
                "Cacoal": "Cacoal",
                "Alta Floresta D'Oeste": "Cacoal",
                "Alto Alegre dos Parecis": "Cacoal",
                "Castanheiras": "Cacoal",
                "Espigão D'Oeste": "Cacoal",
                "Ministro Andreazza": "Cacoal",
                "Nova Brasilândia D'Oeste": "Cacoal",
                "Novo Horizonte do Oeste": "Cacoal",
                "Parecis": "Cacoal",
                "Pimenta Bueno": "Cacoal",
                "Primavera de Rondônia": "Cacoal",
                "Rolim de Moura": "Cacoal",
                "Santa Luzia D'Oeste": "Cacoal",
                "São Felipe D'Oeste": "Cacoal",
                "Vilhena": "Vilhena",
                "Cabixi": "Vilhena",
                "Cerejeiras": "Vilhena",
                "Chupinguaia": "Vilhena",
                "Colorado do Oeste": "Vilhena",
                "Corumbiara": "Vilhena",
                "Pimenteiras do Oeste": "Vilhena",
            }

            df_plotagem_grafico = df_original
            df_plotagem_grafico["regiao_imediata"] = df_plotagem_grafico[
                "munResNome"
            ].map(imediatas)
            df_plotagem_grafico["DTNASC"] = pd.to_datetime(
                df_plotagem_grafico["DTNASC"]
            )
            df_plotagem_grafico["ano_mes"] = df_plotagem_grafico["DTNASC"].dt.to_period(
                "M"
            )
            idade_media = (
                df_plotagem_grafico.groupby(["ano_mes", "regiao_imediata"])["IDADEMAE"]
                .mean()
                .unstack()
            )
            fig, ax = plt.subplots(figsize=(12, 6))
            for regiao in idade_media.columns:
                plt.plot(
                    idade_media.index.to_timestamp(),
                    idade_media[regiao],
                    label=regiao,
                )

            plt.title(
                "Idade Média das Mães ao Longo do Tempo por Região Imediata - Rondônia (2019)"
            )

            plt.xlabel("Tempo")

            plt.ylabel("Idade Média")

            plt.legend(
                title="Região Imediata",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=2,
            )

            plt.grid(True)

            plt.tight_layout()

            buff = BytesIO()
            fig.savefig(buff, format="png", bbox_inches="tight")
            buff.seek(0)

            st.markdown(
                """#### Distribuição da variável `"IDADEMAE"` ao longo de todo o ano""",
                unsafe_allow_html=True,
            )
            st.image(buff, width=1200)

        aba_ifdm, aba_pib = st.tabs(
            [
                "Análise de IFDM - Medição referente ao desenvolvimento do município",
                "Análise de PIB - Medição referente ao valor de produção gerado pela cidade",
            ]
        )
        with aba_ifdm:

            with st.expander("IFDM - Índice Firjan de Desenvolvimento Municipal"):

                @st.cache_data(ttl=86400)
                def fetch_ifdm_data():
                    """Busca e processa dados de IFDM com cache"""
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WIN64; x64)"
                        "AppleWebKit/537.36 (KHTML, like Gecko)"
                        "Chrome/124.0 Safari/537.36"
                    }

                    try:
                        url = "https://pt.wikipedia.org/wiki/Lista_de_munic%C3%ADpios_de_Rond%C3%B4nia_por_IFDM"
                        resp = requests.get(url, headers=headers)
                        resp.raise_for_status()

                        tables = pd.read_html(resp.text)
                        ifdm_table = tables[0]
                        ifdm_table.columns = [
                            "Posição",
                            "Município",
                            "IFDM_Consolidado_2013",
                        ]

                        # Manter como object mesmo - foco na visualização dos dados brutos
                        ifdm_table["Posição"] = ifdm_table["Posição"].astype(str)

                        return ifdm_table

                    except Exception as e:
                        st.error(f"Erro ao carregar dados de IFDM: {e}")
                        return None

                # Carregar dados com cache
                ifdm_table = fetch_ifdm_data()

                if ifdm_table is not None and not ifdm_table.empty:
                    st.success(
                        f"✅ Dados de IFDM carregados com sucesso! {len(ifdm_table)} municípios"
                    )

                    # Layout em colunas
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.subheader("🔍 Seleção do Município")

                        municipios_disponiveis = ifdm_table["Município"].tolist()

                        default_index = 0
                        if "Ariquemes" in municipios_disponiveis:
                            default_index = municipios_disponiveis.index("Ariquemes")
                        elif "Porto Velho" in municipios_disponiveis:
                            default_index = municipios_disponiveis.index("Porto Velho")

                        municipio_de_escolha = st.selectbox(
                            "Selecione o município:",
                            options=municipios_disponiveis,
                            index=default_index,
                            help="Escolha um município para visualizar seus dados de IFDM",
                            key="ifdm_selectbox",
                        )

                        municipio_filtrado = ifdm_table[
                            ifdm_table["Município"] == municipio_de_escolha
                        ]

                        if not municipio_filtrado.empty:
                            municipio_data = municipio_filtrado.iloc[0]

                            st.metric(
                                "🏆 Posição no Ranking", municipio_data["Posição"]
                            )
                            st.metric(
                                "📈 Valor do IFDM (2013)",
                                municipio_data["IFDM_Consolidado_2013"],
                            )

                            # Classificação simplificada baseada no valor string
                            ifdm_str = str(
                                municipio_data["IFDM_Consolidado_2013"]
                            ).replace(",", ".")
                            try:
                                ifdm_val = float(ifdm_str)
                                if ifdm_val >= 0.8:
                                    classificacao = "Alto 🟢"
                                elif ifdm_val >= 0.6:
                                    classificacao = "Moderado 🔵"
                                elif ifdm_val >= 0.4:
                                    classificacao = "Regular 🟠"
                                else:
                                    classificacao = "Baixo 🔴"
                            except:
                                classificacao = "Classificação indisponível ⚪"

                            st.metric("🎯 Classificação", classificacao)

                        else:
                            st.warning(
                                f"⚠️ Dados não encontrados para {municipio_de_escolha}"
                            )

                    with col2:
                        st.subheader("📊 Ranking Completo de IFDM")

                        # Mostrar tabela simples sem formatação complexa
                        st.dataframe(
                            ifdm_table,
                            use_container_width=True,
                            height=400,
                            hide_index=True,
                        )

                    # Análise adicional simplificada
                    st.subheader("📈 Visualização dos Dados")

                    tab1, tab2 = st.tabs(["Dados Completos", "Informações"])

                    with tab1:
                        st.write("**Tabela completa de IFDM por município:**")
                        st.dataframe(ifdm_table, use_container_width=True)

                    with tab2:
                        st.write(
                            """
                            **Sobre o IFDM:**
                            - Índice que varia de 0 a 1
                            - Classificação:
                            * 0.8 - 1.0: Alto 🟢
                            * 0.6 - 0.8: Moderado 🔵  
                            * 0.4 - 0.6: Regular 🟠
                            * 0.0 - 0.4: Baixo 🔴
                            """
                        )
                        st.write(f"**Total de municípios:** {len(ifdm_table)}")
                        st.write(
                            "**Fonte:** Wikipedia - Lista de municípios de Rondônia por IFDM"
                        )

                else:
                    st.warning("⚠️ Não foi possível carregar os dados de IFDM.")
        with aba_pib:
            with st.expander("PIB - Produto Interno Bruto"):

                @st.cache_data(ttl=86400)  # Adicionei TTL aqui também
                def fetch_pib_data():
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WIN64; x64)"
                        "AppleWebKit/537.36 (KHTML, like Gecko)"
                        "Chrome/124.0 Safari/537.36"
                    }
                    try:
                        url = "https://pt.wikipedia.org/wiki/Lista_de_munic%C3%ADpios_de_Rond%C3%B4nia_por_PIB"
                        resp = requests.get(url, headers=headers)
                        resp.raise_for_status()

                        tables = pd.read_html(resp.text)
                        pib_table = tables[0]

                        colunas_selecionadas = tables[0].columns.get_level_values(1)

                        # Renomear colunas para garantir consistência
                        pib_table.columns = colunas_selecionadas

                        # Limpeza do PIB - remover "R$", espaços, etc.
                        pib_table["PIB"] = (
                            pib_table["PIB"]
                            .astype(str)
                            .str.replace("R\$", "", regex=False)
                            .str.replace(
                                ".", "", regex=False
                            )  # Remove pontos de milhares
                            .str.replace(
                                ",", ".", regex=False
                            )  # Converte vírgula decimal para ponto
                            .str.replace(" ", "", regex=False)
                        )

                        # Converter para numérico
                        pib_table["PIB"] = pd.to_numeric(
                            pib_table["PIB"], errors="coerce"
                        )

                        # Ordenar o DataFrame pelo PIB em ordem decrescente
                        pib_table = pib_table.sort_values("PIB", ascending=False)

                        # Criar a coluna de posição (ranking) baseada na ordem
                        pib_table["Posição"] = range(1, len(pib_table) + 1)

                        return pib_table

                    except Exception as e:
                        st.error(f"Erro ao carregar os dados do PIB: {e}")
                        return None

                # CORREÇÃO: Esta linha estava no lugar errado
                pib_table = fetch_pib_data()

                if pib_table is not None and not pib_table.empty:
                    st.success(
                        f"✅ Dados de PIB carregados com sucesso! {len(pib_table)} municípios"
                    )

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.subheader("🔍 Seleção do Município")

                        municipios_disponiveis_pib = pib_table["Município"].tolist()

                        default_index = 0
                        if "Ariquemes" in municipios_disponiveis_pib:
                            default_index = municipios_disponiveis_pib.index(
                                "Ariquemes"
                            )
                        elif "Porto Velho" in municipios_disponiveis_pib:
                            default_index = municipios_disponiveis_pib.index(
                                "Porto Velho"
                            )

                        municipio_de_escolha_pib = st.selectbox(
                            "Selecione o município:",
                            options=municipios_disponiveis_pib,
                            index=default_index,
                            help="Escolha um município para visualizar seus dados sobre o PIB",
                            key="pib_selectbox",
                        )

                        municipio_filtrado_pib = pib_table[
                            pib_table["Município"] == municipio_de_escolha_pib
                        ]

                        if not municipio_filtrado_pib.empty:
                            pib_municipio_data = municipio_filtrado_pib.iloc[0]

                            st.metric(
                                "🏆 Posição no Ranking",
                                str(pib_municipio_data["Posição"]),
                            )

                            # Formatar PIB em milhões/bilhões
                            pib_valor = pib_municipio_data["PIB"]
                            if pd.notna(pib_valor):
                                if pib_valor >= 1_000_000_000:
                                    pib_formatado = (
                                        f"R$ {pib_valor/1_000_000_000:.2f} bilhões"
                                    )
                                else:
                                    pib_formatado = (
                                        f"R$ {pib_valor/1_000_000:.2f} milhões"
                                    )
                                st.metric("📈 Valor do PIB", pib_formatado)
                            else:
                                st.metric("📈 Valor do PIB", "Dado indisponível")

                            # Classificação do PIB
                            if pd.notna(pib_valor):
                                if pib_valor > 1_000_000_000:
                                    classificacao_pib = "Acima de 1 bilhão 🟢"
                                elif pib_valor > 500_000_000:
                                    classificacao_pib = "500 milhões - 1 bilhão 🔵"
                                elif pib_valor > 300_000_000:
                                    classificacao_pib = "300 - 500 milhões 🟠"
                                elif pib_valor > 200_000_000:
                                    classificacao_pib = "200 - 300 milhões 🟡"
                                elif pib_valor > 100_000_000:
                                    classificacao_pib = "100 - 200 milhões 🟣"
                                else:
                                    classificacao_pib = "Até 100 milhões 🔴"
                            else:
                                classificacao_pib = "Classificação indisponível ⚪"

                            st.metric("🎯 Classificação", classificacao_pib)

                        else:
                            st.warning(
                                f"⚠️ Dados não encontrados para {municipio_de_escolha_pib}"
                            )

                    with col2:
                        st.subheader("📊 Ranking Completo de PIB")

                        # Mostrar tabela com PIB formatado
                        pib_table_display = pib_table.copy()
                        pib_table_display["PIB Formatado"] = pib_table_display[
                            "PIB"
                        ].apply(
                            lambda x: (
                                f"R$ {x/1_000_000:,.2f} mi" if pd.notna(x) else "N/A"
                            )
                        )

                        st.dataframe(
                            pib_table_display[
                                ["Posição", "Município", "PIB Formatado"]
                            ],
                            use_container_width=True,
                            height=400,
                            hide_index=True,
                        )

                else:
                    st.warning("⚠️ Não foi possível carregar os dados de PIB.")

    # ------------------------------------------------------
    # PONTO DE ENTRADA
    # ------------------------------------------------------


if __name__ == "__main__":
    main()
