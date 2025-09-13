import pandas as pd
import streamlit as st
import requests
import numpy as np
import time
from io import StringIO

# Configura o título da aplicação
st.set_page_config(
    page_title="Database de nascimentos em Roraima 2019",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="varig_icon.png",
)


def main():

    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">
        <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 2.5em;'>
            <strong>🔍 Análise Exploratória - SINASC Roraima 2019</strong>
        </h1>
        <p style='text-align: center; font-family: "Kantumruy Pro", sans-serif;'>
            Diagnóstico completo da base de nascimentos do Sistema de Informações sobre Nascidos Vivos
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Carregamento dos dados com feedback
    with st.spinner("Carregando base de dados SINASC..."):
        df = pd.read_csv("./output/SINASC_RO_2019.csv", keep_default_na=True)

    st.success(
        f"✅ Base carregada com sucesso: {df.shape[0]} linhas e {df.shape[1]} colunas"
    )

    if "df_feature_selected" not in st.session_state:
        st.session_state.df_feature_selected = df[
            [
                "LOCNASC",
                "IDADEMAE",
                "ESTCIVMAE",
                "ESCMAE",
                "QTDFILVIVO",
                "GESTACAO",
                "GRAVIDEZ",
                "CONSULTAS",
                "APGAR5",
            ]
        ]
    df_feature_selected = st.session_state.df_feature_selected

    # No topo, após inicializar df_feature_selected
    if "should_rerun" not in st.session_state:
        st.session_state.should_rerun = False

    # Expander principal de diagnóstico
    with st.expander("📊 Diagnóstico Completo da Base de Dados", expanded=True):

        # Seção 1: Visão Geral
        st.subheader("📈 Visão Geral da Base")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Registros", df.shape[0])
        with col2:
            st.metric("Total de Colunas", df.shape[1])
        with col3:
            st.metric(
                "Tamanho da Memória",
                f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            )
        # Seção 2: Duplicatas
        st.subheader("🔍 Análise de Duplicatas")
        duplicatas_completas = df.duplicated().sum()
        duplicatas_parciais = (
            df.duplicated(subset=["DTNASC", "SEXO", "PESO"]).sum()
            if all(col in df.columns for col in ["DTNASC", "SEXO", "PESO"])
            else 0
        )

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Duplicatas completas:** {duplicatas_completas}")
        with col2:
            st.info(f"**Possíveis duplicatas (chave parcial):** {duplicatas_parciais}")

        if duplicatas_completas > 0:
            st.warning("⚠️ Foram encontradas linhas duplicadas completas na base")
            if st.button("Visualizar linhas duplicadas"):
                st.dataframe(
                    df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist())
                )

        # Seção 3: Diagnóstico de valores nulos
        st.subheader("❌ Análise de Valores Nulos")

        nulos_absolutos = df.isna().sum().sort_values(ascending=False)
        nulos_percentual = (df.isna().sum() / len(df) * 100).sort_values(
            ascending=False
        )
        nulos_df = pd.DataFrame(
            {
                "Coluna": nulos_absolutos.index,
                "Valores Nulos": nulos_absolutos.values,
                "Percentual_Nulos": nulos_percentual.values.round(2),
            }
        )

        colunas_com_nulos = nulos_df[nulos_df["Valores Nulos"] > 0]

        exibicao_nulos = st.radio(
            "Selecione formato da exibição dos valores nulos",
            ["Números absolutos", "Número Percentual"],
            horizontal=True,
        )

        if exibicao_nulos == "Números absolutos":
            st.dataframe(
                colunas_com_nulos[["Coluna", "Valores Nulos"]].rename(
                    columns={"Valores Nulos": "Qtd. Valores Nulos"}
                ),
                use_container_width=True,
                height=400,
            )
        if exibicao_nulos == "Número Percentual":
            # Gráfico de barras + tabela
            col1, col2 = st.columns([2, 1])

            with col1:
                st.bar_chart(
                    data=colunas_com_nulos.set_index("Coluna")["Percentual_Nulos"],
                    color="#ff4b4b",  # Cor temática para alerta
                )

            with col2:
                st.dataframe(
                    colunas_com_nulos[["Coluna", "Percentual_Nulos"]].rename(
                        columns={"Percentual_Nulos": "% Nulos"}
                    ),
                    use_container_width=True,
                    height=400,
                )
        st.subheader("✂️ Feature Selection")
        col_df_all_columns, col_df_features_selected = st.columns([1, 1])
        st.caption(
            """* Nesta etapa iremos deixar apenas as variáveis consideradas relevantes pelas orientações do tutor, sendo elas: ` 
['LOCNASC', 'IDADEMAE', 'ESTCIVMAE', 'ESCMAE', 'QTDFILVIVO', 
    'GESTACAO', 'GRAVIDEZ', 'CONSULTAS', 'APGAR5'] 
`"""
        )

        if st.checkbox(
            "Exibir valores nulos para dataframe com variáveis selecionadas"
        ):
            nulos_absolutos = (
                df_feature_selected.isna().sum().sort_values(ascending=False)
            )
            nulos_percentual = (
                df_feature_selected.isna().sum() / len(df_feature_selected) * 100
            ).sort_values(ascending=False)
            nulos_df_feature_selected = pd.DataFrame(
                {
                    "Coluna": nulos_absolutos.index,
                    "Valores Nulos": nulos_absolutos.values,
                    "Percentual_Nulos": nulos_percentual.values.round(2),
                }
            )

            colunas_com_nulos_feature_selected = nulos_df_feature_selected[
                nulos_df_feature_selected["Valores Nulos"] > 0
            ]

            exibicao_nulos_2 = st.radio(
                "Selecione formato da exibição dos valores nulos",
                ["Números absolutos", "Número Percentual"],
                horizontal=True,
                key="exibicao_nulos_df_feature_selected",
            )

            if exibicao_nulos_2 == "Números absolutos":
                st.dataframe(
                    colunas_com_nulos_feature_selected[
                        ["Coluna", "Valores Nulos"]
                    ].rename(columns={"Valores Nulos": "Qtd. Valores Nulos"}),
                    use_container_width=True,
                    height=400,
                )
            if exibicao_nulos_2 == "Número Percentual":
                # Gráfico de barras + tabela
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.bar_chart(
                        data=colunas_com_nulos_feature_selected.set_index("Coluna")[
                            "Percentual_Nulos"
                        ],
                        color="#ff4b4b",  # Cor temática para alerta
                    )

                with col2:
                    st.dataframe(
                        colunas_com_nulos_feature_selected[
                            ["Coluna", "Percentual_Nulos"]
                        ].rename(columns={"Percentual_Nulos": "% Nulos"}),
                        use_container_width=True,
                        height=400,
                    )
        aba_1, aba_2, aba_3, aba_4 = st.tabs(
            [
                "Remoção de linhas ausentes da variável `APGAR5`",
                "Preenchendo as variáveis `ESTCIVMAE` e `CONSULTAS`",
                "Substituir os valores ausentes de `QTDFILVIVO` por zero",
                "Deletando as linhas ausentes remanscentes",
            ]
        )

        with aba_1:
            if st.button("""Remover dados faltantes da variável "APGAR5" """):
                # 1. Primeiro, mostre o que será removido (para transparência)
                linhas_para_remover = st.session_state.df_feature_selected[
                    df_feature_selected["APGAR5"].isna()
                ]
                st.info(
                    f"ℹ️ Serão removidas {len(linhas_para_remover)} linhas com APGAR5 faltante"
                )

                if not linhas_para_remover.empty:
                    st.dataframe(
                        linhas_para_remover.head()
                    )  # Mostra amostra do que será removido

                # 2. AGORA SIM: Remova efetivamente as linhas e atualize o estado
                # Mantenha apenas as linhas onde APGAR5 NÃO é nulo
                st.session_state.df_feature_selected = (
                    st.session_state.df_feature_selected.dropna(subset=["APGAR5"])
                )

                # 3. Confirmação da ação
                st.success(
                    f"✅ {len(linhas_para_remover)} linhas com APGAR5 faltante removidas com sucesso!"
                )
                st.metric(
                    "Linhas restantes na base",
                    len(st.session_state.df_feature_selected),
                )

                # Feedback visual com temporizador
                success_placeholder = st.empty()
                success_placeholder.success(
                    f"✅ {len(linhas_para_remover)} linhas removidas! Atualizando em 2 segundos..."
                )

                # Timer apenas para o feedback visual
                for i in range(2, 0, -1):
                    success_placeholder.success(
                        f"✅ {len(linhas_para_remover)} linhas removidas! Atualizando em {i} segundos..."
                    )
                    time.sleep(1)

                success_placeholder.empty()
                st.session_state.should_rerun = True

        with aba_2:
            if st.button(
                """Preencher dados faltantes da variável "ESTCIVMAE" e "CONSULTAS" """
            ):
                linhas_para_preencher = st.session_state.df_feature_selected[
                    df_feature_selected["ESTCIVMAE"].isna()
                    | df_feature_selected["CONSULTAS"].isna()
                ]
                st.info(
                    f"""Serão preenchidas {len(linhas_para_preencher)} com o número 9 que possuí o significado de "Ignorado" neste contexto"""
                )

                if not linhas_para_preencher.empty:
                    st.dataframe(linhas_para_preencher.head())

                st.session_state.df_feature_selected["ESTCIVMAE"] = (
                    st.session_state.df_feature_selected["ESTCIVMAE"].fillna(9)
                )
                st.session_state.df_feature_selected["CONSULTAS"] = (
                    st.session_state.df_feature_selected["CONSULTAS"].fillna(9)
                )

                # Feedback visual com temporizador
                success_placeholder = st.empty()
                success_placeholder.success(
                    f"✅ {len(linhas_para_preencher)} linhas preenchidas! Atualizando em 2 segundos..."
                )

                # Timer apenas para o feedback visual
                for i in range(2, 0, -1):
                    success_placeholder.success(
                        f"✅ {len(linhas_para_preencher)} linhas preenchidas! Atualizando em {i} segundos..."
                    )
                    time.sleep(1)

                success_placeholder.empty()
                st.session_state.should_rerun = True
        with aba_3:
            if st.button(
                """Substituir dados faltantes da variável "QTDFILVIVO" por zero """
            ):
                linhas_para_preencher = st.session_state.df_feature_selected[
                    df_feature_selected["QTDFILVIVO"].isna()
                ]
                st.info(
                    f"""Serão preenchidas {len(linhas_para_preencher)} com o número 9 que possuí o significado de "Ignorado" neste contexto"""
                )

                if not linhas_para_preencher.empty:
                    st.dataframe(linhas_para_preencher.head())

                st.session_state.df_feature_selected["QTDFILVIVO"] = (
                    st.session_state.df_feature_selected["QTDFILVIVO"].fillna(0)
                )

                # Feedback visual com temporizador
                success_placeholder = st.empty()
                success_placeholder.success(
                    f"✅ {len(linhas_para_preencher)} linhas preenchidas! Atualizando em 2 segundos..."
                )

                # Timer apenas para o feedback visual
                for i in range(2, 0, -1):
                    success_placeholder.success(
                        f"✅ {len(linhas_para_preencher)} linhas preenchidas! Atualizando em {i} segundos..."
                    )
                    time.sleep(1)

                success_placeholder.empty()
                st.session_state.should_rerun = True

        with aba_4:
            if st.button("""Excluir linhas remanescentes com valores ausentes"""):
                # CORREÇÃO: Usar any(axis=1) para encontrar linhas com qualquer NaN
                linhas_para_remover_2 = st.session_state.df_feature_selected[
                    st.session_state.df_feature_selected.isna().any(axis=1)
                ]

                st.info(
                    f"ℹ️ Serão removidas {len(linhas_para_remover_2)} linhas com valores ausentes"
                )

                if not linhas_para_remover_2.empty:
                    st.dataframe(linhas_para_remover_2.head())

                # Remover linhas com qualquer valor NaN
                st.session_state.df_feature_selected = (
                    st.session_state.df_feature_selected.dropna()
                )

                # Feedback visual
                success_placeholder = st.empty()
                success_placeholder.success(
                    f"✅ {len(linhas_para_remover_2)} linhas removidas! Atualizando em 2 segundos..."
                )

                for i in range(2, 0, -1):
                    success_placeholder.success(
                        f"✅ {len(linhas_para_remover_2)} linhas removidas! Atualizando em {i} segundos..."
                    )
                    time.sleep(1)

                success_placeholder.empty()
                st.session_state.should_rerun = True

    # MOVER a categorização APGAR5 para dentro de um controle
    if st.button("Categorizar APGAR5"):  # ← Adicione um botão para isso
        st.session_state.df_feature_selected.loc[
            (st.session_state.df_feature_selected["APGAR5"] >= 8)
            & (st.session_state.df_feature_selected["APGAR5"] <= 10),
            "CAT_APGAR5",
        ] = "normal"

        st.session_state.df_feature_selected.loc[
            (st.session_state.df_feature_selected["APGAR5"] >= 6)
            & (st.session_state.df_feature_selected["APGAR5"] <= 7),
            "CAT_APGAR5",
        ] = "asfixia leve"

        st.session_state.df_feature_selected.loc[
            (st.session_state.df_feature_selected["APGAR5"] >= 4)
            & (st.session_state.df_feature_selected["APGAR5"] <= 5),
            "CAT_APGAR5",
        ] = "asfixia moderada"

        st.session_state.df_feature_selected.loc[
            (st.session_state.df_feature_selected["APGAR5"] >= 0)
            & (st.session_state.df_feature_selected["APGAR5"] <= 3),
            "CAT_APGAR5",
        ] = "asfixia severa"

        snake_case = lambda texto_string: str(texto_string.upper().replace(" ", "_"))
        st.session_state.df_feature_selected["CAT_APGAR5"] = (
            st.session_state.df_feature_selected["CAT_APGAR5"].apply(snake_case)
        )

        st.session_state.should_rerun = True

    st.subheader("Visualização final do dataframe")

    # Filtro por categoria APGAR5
    categorias_apgar = sorted(
        st.session_state.df_feature_selected["CAT_APGAR5"].unique()
    )
    categoria_selecionada = st.multiselect(
        "Selecione as categorias de APGAR5:",
        options=categorias_apgar,
        default=categorias_apgar,  # Seleciona todas por padrão
    )

    # Filtro adicional por faixa etária da mãe
    idade_min, idade_max = st.slider(
        "Faixa etária da mãe:",
        min_value=int(st.session_state.df_feature_selected["IDADEMAE"].min()),
        max_value=int(st.session_state.df_feature_selected["IDADEMAE"].max()),
        value=(
            int(st.session_state.df_feature_selected["IDADEMAE"].min()),
            int(st.session_state.df_feature_selected["IDADEMAE"].max()),
        ),
    )

    # Aplicar os filtros
    dataframe_filtrado = st.session_state.df_feature_selected[
        (st.session_state.df_feature_selected["CAT_APGAR5"].isin(categoria_selecionada))
        & (st.session_state.df_feature_selected["IDADEMAE"] >= idade_min)
        & (st.session_state.df_feature_selected["IDADEMAE"] <= idade_max)
    ]

    # Mostrar estatísticas
    st.success(
        f"**{len(dataframe_filtrado)} registros** correspondem aos filtros aplicados"
    )

    # Mostrar dataframe
    st.dataframe(dataframe_filtrado)

    # Gráfico de distribuição
    if len(categoria_selecionada) > 0:
        st.bar_chart(dataframe_filtrado["CAT_APGAR5"].value_counts())

    if st.session_state.should_rerun:
        st.session_state.should_rerun = False  # Reseta a flag
        st.rerun()


if __name__ == "__main__":
    main()
