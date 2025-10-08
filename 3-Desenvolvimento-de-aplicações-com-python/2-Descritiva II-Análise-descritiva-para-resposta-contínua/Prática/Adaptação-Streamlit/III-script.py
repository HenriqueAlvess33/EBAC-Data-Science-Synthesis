import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO


# ------------------------------------------------------
# FUNÇÃO PARA CARREGAR DADOS
# ------------------------------------------------------


# 🚀 ADICIONAR CACHE PARA PERFORMANCE
@st.cache_data
def load_data(conjunto_de_dados):
    """
    Versão com cache para não recarregar os dados a cada interação.
    """
    try:
        return pd.read_csv(conjunto_de_dados)
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return pd.DataFrame()  # Retorna DataFrame vazio para evitar quebras


# 🎯 FUNÇÃO MELHORADA PARA SELEÇÃO DE DATAS
def criar_widget_temporal(previsao_renda, key_suffix=""):
    """
    Versão mais flexível que permite seleção por mês, trimestre ou ano.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        anos = sorted(previsao_renda["data_ref"].dt.year.unique())
        ano_selecionado = st.selectbox("Ano", anos, key=f"ano_{key_suffix}")

    with col2:
        meses = sorted(
            previsao_renda[previsao_renda["data_ref"].dt.year == ano_selecionado][
                "data_ref"
            ].dt.month.unique()
        )
        meses_portugues = {
            1: "Janeiro",
            2: "Fevereiro",
            3: "Março",
            4: "Abril",
            5: "Maio",
            6: "Junho",
            7: "Julho",
            8: "Agosto",
            9: "Setembro",
            10: "Outubro",
            11: "Novembro",
            12: "Dezembro",
        }  # seu mapping

        mes_selecionado = st.selectbox(
            "Mês",
            meses,
            format_func=lambda x: meses_portugues[x],
            key=f"mes_{key_suffix}",
        )

    return ano_selecionado, mes_selecionado


def selecao_mes(previsao_renda, key):
    """
    Função auxiliar que constrói um selectbox para seleção de mês.
    - Recebe o DataFrame `previsao_renda` e uma `key` para o widget.
    - Mapeia números de mês para nomes em português (Janeiro...Dezembro).
    - Exibe um st.selectbox com as opções de mês existentes em previsao_renda["mês"].
    - Retorna o valor numérico do mês selecionado (1..12).
    """
    meses_portugues = {
        1: "Janeiro",
        2: "Fevereiro",
        3: "Março",
        4: "Abril",
        5: "Maio",
        6: "Junho",
        7: "Julho",
        8: "Agosto",
        9: "Setembro",
        10: "Outubro",
        11: "Novembro",
        12: "Dezembro",
    }
    # selectbox com formatação do rótulo via format_func para mostrar nomes dos meses
    selecao_de_data = st.selectbox(
        label="Selecione a data que deseja visualizar a distribuição de categorias",
        options=sorted(
            previsao_renda["mês"].unique()
        ),  # opções: meses presentes no dataset
        format_func=lambda x: meses_portugues[
            x
        ],  # converte número -> nome do mês na exibição
        key=key,
    )

    return selecao_de_data  # retorna o número do mês selecionado


# ------------------------------------------------------
# CONFIGURAÇÃO INICIAL DA PÁGINA STREAMLIT
# ------------------------------------------------------
st.set_page_config(
    page_title="Análise das variáveis qualitativas",  # Título da aba do navegador
    layout="wide",  # Usa toda a largura da tela (layout expandido)
    initial_sidebar_state="expanded",  # Abre a sidebar por padrão
    page_icon="varig_icon.png",  # Ícone da aplicação (arquivo local esperado)
)


def main():
    """
    Função principal que organiza os widgets e gráficos do app.
    - Faz upload, limpeza mínima, seleção de variáveis qualitativas,
      visualizações de frequência ao longo do tempo e análise de renda por categoria.
    """

    # --------------------------------------------------
    # UPLOAD DO DATASET (BARRA LATERAL)
    # --------------------------------------------------
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Faça o upload do dataset `previsao_renda.csv`",
            type="csv",  # apenas CSV conforme a sua versão do script
            help="Somente serão aceitos arquivos CSV",
        )

    # Se um arquivo foi enviado pelo usuário, processa-o
    if uploaded_file is not None:
        st.sidebar.success("Dataset carregado")  # feedback ao usuário
        previsao_renda = load_data(uploaded_file)  # chama função de leitura

        # --------------------------------------------------
        # REMOÇÃO DE COLUNAS E PREPARAÇÃO BÁSICA
        # --------------------------------------------------
        colunas_existentes = ["Unnamed: 0", "mau", "index"]
        # Gera lista apenas com as colunas que existem no DataFrame
        colunas_para_remover = [
            col for col in colunas_existentes if col in previsao_renda.columns
        ]

        # Remove in-place as colunas indesejadas (se existirem)
        previsao_renda.drop(columns=colunas_para_remover, inplace=True)

        # Converte a coluna 'data_ref' para datetime para permitir extração de mês
        previsao_renda["data_ref"] = pd.to_datetime(previsao_renda["data_ref"])
        # Cria coluna numérica 'mês' extraída de data_ref (1..12)
        previsao_renda["mês"] = previsao_renda["data_ref"].dt.month

        # --------------------------------------------------
        # SELEÇÃO DE VARIÁVEIS QUALITATIVAS (TIPO OBJECT)
        # --------------------------------------------------
        # Seleciona todas as colunas do tipo object (strings / categorias)
        colunas_para_selectbox = previsao_renda.select_dtypes(
            include=["object"]
        ).columns

        # Widget para escolher qual coluna qualitativa será analisada nos gráficos
        variavel_x = st.selectbox(
            options=colunas_para_selectbox,
            label="Selecione a coluna que irá aparecer no gráfico",
        )

        # --------------------------------------------------
        # CRIAÇÃO DE ABAS PARA AS DUAS ANÁLISES
        # --------------------------------------------------
        tab_1, tab_2 = st.tabs(
            [
                "Analisando a frequência das categorias ao longo do tempo",
                "Analisando o desempenho de cada categoria na variável `renda` ",
            ]
        )

        # ------------------ ABA 1: Frequência ao longo do tempo ------------------
        with tab_1:
            st.write(f"#### Entendimento da variável `{variavel_x}`")
            # Divide em duas colunas: esquerda (gráfico), direita (tabela de contagem)
            col_1, col_2 = st.columns(2)
            with col_1:
                # Cria figura matplotlib e eixo para o countplot
                fig, ax = plt.subplots(figsize=(12, 7))
                # Plota countplot com hue para separar categorias da variável selecionada
                sns.countplot(x="data_ref", hue=variavel_x, data=previsao_renda, ax=ax)

                # Preparação dos rótulos do eixo x: formatar as datas para "%m-%Y"
                tick_labs = (
                    previsao_renda["data_ref"]
                    .map(lambda ts: ts.strftime("%m-%Y"))
                    .unique()
                )
                # Define ticks no eixo x com base na quantidade de datas únicas
                ticks = ax.set_xticks(list(range(previsao_renda["data_ref"].nunique())))
                # Substitui os rótulos pelo formato mês-ano e rotaciona para legibilidade
                labels = ax.set_xticklabels(tick_labs, rotation=45)

                # Posiciona a legenda fora do gráfico para não sobrepor conteúdo
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                # Título do gráfico
                ax.set_title(f"Distribuição da variável {variavel_x}")
                # Ajusta layout para evitar cortes de labels/legenda
                plt.tight_layout()

                # Salva a figura em memória e exibe como imagem no Streamlit
                buff = BytesIO()
                fig.savefig(buff, format="png", bbox_inches="tight")
                buff.seek(0)
                st.image(buff, width=1200)

            with col_2:
                # Usa a função auxiliar para selecionar mês (retorna número do mês)
                selecao_de_data = selecao_mes(previsao_renda, key="selecao_mes_aba_1")

                # Conta as ocorrências da variável selecionada apenas no mês escolhido
                contagem_data_selecionada = pd.DataFrame(
                    previsao_renda.loc[
                        previsao_renda["mês"] == selecao_de_data, variavel_x
                    ].value_counts()
                )
                # Mostra a tabela de contagem ao usuário
                st.dataframe(contagem_data_selecionada)

        # ------------------ ABA 2: Desempenho das categorias em relação à renda ------------------
        with tab_2:
            # Duas colunas para seleção: categoria e mês
            coluna_selecao_1, coluna_selecao_2 = st.columns(2)
            with coluna_selecao_1:
                # Selectbox com todas as categorias possíveis da variável escolhida
                selecao_categoria = st.selectbox(
                    label="Selecione a categoria avaliada",
                    options=previsao_renda[variavel_x].value_counts().index,
                )
            with coluna_selecao_2:
                # Seleciona o mês através da função auxiliar (segunda aba)
                selecao_de_data_aba_2 = selecao_mes(
                    previsao_renda, key="selecao_mes_aba_2"
                )

            # Linha divisória visual
            st.divider()

            # Área principal dividida: coluna menor para gráfico (1) e coluna maior para métricas (2)
            coluna_principal_1, colcoluna_principal_2, coluna_principal_3 = st.columns(
                [2, 1, 2]
            )

            with coluna_principal_1:
                st.write(
                    f"#### Média de renda da variável {variavel_x} ao longo do tempo"
                )
                # Gera figura com pointplot (média de renda por data, separada por categoria)
                fig_2, ax_2 = plt.subplots(figsize=(12, 7))
                sns.pointplot(
                    x="data_ref",
                    y="renda",
                    hue=variavel_x,
                    data=previsao_renda,
                    ax=ax_2,
                    dodge=True,
                    errorbar=("ci", 95),  # intervalo de confiança 95%
                )

                # Prepara rótulos do eixo x no formato mês-ano
                tick_labs = (
                    previsao_renda["data_ref"]
                    .map(lambda ts: ts.strftime("%m-%Y"))
                    .unique()
                )
                ticks_two = ax_2.set_xticks(
                    list(range(previsao_renda["data_ref"].nunique()))
                )
                labels_two = ax_2.set_xticklabels(tick_labs, rotation=45)

                # Ajustes de legenda, título e layout
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                ax.set_title(f"Distribuição da variável {variavel_x}")
                plt.tight_layout()

                # Salva e exibe figura como imagem
                buff_two = BytesIO()
                fig_2.savefig(buff_two, format="png", bbox_inches="tight")
                buff_two.seek(0)
                st.image(buff_two, width=1200)

            # Coluna à direita: métricas numéricas resumidas para a categoria selecionada
            with coluna_principal_3:
                st.write("#### Dados das médias de renda presentes nos dados")

                # Calcula a renda média da categoria selecionada ao longo de todo o período
                renda_media_ano = (
                    previsao_renda.loc[
                        previsao_renda[variavel_x] == selecao_categoria, "renda"
                    ]
                    .mean()
                    .round(2)
                )
                st.metric(
                    f"Renda Média da categoria no ano",
                    renda_media_ano,
                )

                # Calcula a renda média da categoria no mês selecionado
                renda_media_categoria_mes = (
                    previsao_renda.loc[
                        (previsao_renda["mês"] == selecao_de_data_aba_2)
                        & (previsao_renda[variavel_x] == selecao_categoria),
                        "renda",
                    ]
                    .mean()
                    .round(2)
                )
                st.metric(
                    f"Renda Média da categoria no mês:",
                    renda_media_categoria_mes,
                )

                # Renda média global (todas as categorias)
                renda_media_variavel = previsao_renda["renda"].mean().round(2)
                st.metric(
                    f"Renda Média de todos os públicos",
                    renda_media_variavel,
                )

                # Diferença entre a renda média da categoria e a renda média geral
                diferenca_renda_media_categoria_x_variavel = round(
                    (renda_media_ano - renda_media_variavel), 2
                )
                st.metric(
                    f"Média da categoria - Média Total",
                    diferenca_renda_media_categoria_x_variavel,
                )


# ------------------------------------------------------
# PONTO DE ENTRADA
# ------------------------------------------------------
if __name__ == "__main__":
    main()  # Executa a função principal ao rodar o script

    # informacoes_para_o_grafico = st.radio(
    #     "Categorias a serem apresentadas nos gráficos",
    #     ["Todas", "Selecionadas"],
    #     horizontal=True,
    # )
