import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Criação de um identificador para bons pagadores e maus pagadores",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="varig_icon.png",
)


def main():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">
        <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 2.5em;'>
            <strong>Criando um marcador de maus pagadores á partir de uma base de pagamentos</strong>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Carregando base de dados"):
        propostas = pd.read_csv("./output/application_record.csv")
        pg = pd.read_csv("./output/pagamentos_largo.csv")
    st.success(
        f"""
    ✅ **Base de propostas:** {propostas.shape[0]} linhas e {propostas.shape[1]} colunas  
    ✅ **Base de pagamentos:** {pg.shape[0]} linhas e {pg.shape[1]} colunas
    """
    )

    with st.expander("Apresentação dos Dataframes"):
        st.subheader("Dataframe de pagamentos e de propostas:")
        selecao_de_dataframe = st.radio(
            "Selecione qual será o dataframe apresentado",
            ["Propostas", "Pagamentos"],
            horizontal=True,
            key="selecao_dataframe_apresentacao_radio",
        )

        if selecao_de_dataframe == "Propostas":
            st.dataframe(propostas)
        else:
            aba_dataframe_completo, aba_value_counts = st.tabs(
                ["Dataframe completo", "Quantidade de atrasos x Não atrasos"],
            )
            with aba_dataframe_completo:
                st.dataframe(pg)
            with aba_value_counts:
                # Sua análise atual
                verificacao_default = pg.isin([2, 3, 4, 5])
                valores = verificacao_default.stack()

                crosstab_total = pd.crosstab(index=valores, columns="Contagem").reindex(
                    [False, True], fill_value=0
                )
                crosstab_total.index = ["Bons Pagadores", "Maus Pagadores"]

                st.bar_chart(crosstab_total)
                st.dataframe(crosstab_total)
                # Adicionar métricas mais detalhadas
                st.subheader("📈 Análise Detalhada por Status")

                # Contagem por cada código de status
                status_counts = pd.DataFrame()
                for status in [0, 1, 2, 3, 4, 5, "C"]:
                    count = (
                        (pg == status).sum().sum()
                    )  # Soma todos os valores no dataframe
                    status_counts.loc[status, "Contagem"] = count

                status_counts.index.name = "Código Status"
                st.dataframe(status_counts)

                # Interpretação dos códigos (baseado no padrão comum)
                st.info(
                    """
                **Interpretação comum dos códigos:**
                - 0: 1-29 dias em atraso (Past Due) 
                - 1: 30-59 dias em atraso (Overdue)
                - 2: 60-89 dias em atraso (Overdue Grave)
                - 3: 90-119 dias em atraso (Overdue Grave)
                - 4: 120-149 dias em atraso (Overdue Grave)
                - 5: +150 dias em atraso (Overdue Grave)
                - C: Pagou em dia (Adimplente)
                """
                )
    # Adicione esta seção após sua análise atual
    with st.expander("🎯 Criação do Identificador de Maus Pagadores"):
        st.subheader("Definindo a Variável Target")

        # Criar variável target consolidada
        pg["mau_pagador"] = pg.isin([2, 3, 4, 5]).any(axis=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            total_clientes = len(pg)
            st.metric("Total de Clientes", total_clientes)
        with col2:
            maus_pagadores = pg["mau_pagador"].sum()
            st.metric("Maus Pagadores", maus_pagadores)
        with col3:
            taxa_maus = (
                (maus_pagadores / total_clientes * 100) if total_clientes > 0 else 0
            )
            st.metric("Taxa de Maus Pagadores", f"{taxa_maus:.1f}%")

        # Gráfico de distribuição
        dist_target = pg["mau_pagador"].value_counts()

        dist_target = pg["mau_pagador"].value_counts()

        # Verificar quais valores existem antes de renomear
        if len(dist_target) == 2:
            dist_target.index = ["Bons Pagadores", "Maus Pagadores"]
        elif len(dist_target) == 1:
            if dist_target.index[0] == True:
                dist_target.index = ["Maus Pagadores"]
            else:
                dist_target.index = ["Bons Pagadores"]
        st.bar_chart(dist_target)

        st.success("✅ Variável target 'mau_pagador' criada com sucesso!")

        # Unir com dados demográficos para análise mais rica
    if "ID" in propostas.columns and "ID" in pg.columns:
        dados_completos = pd.merge(
            propostas, pg[["ID", "mau_pagador"]], on="ID", how="inner"
        )


if __name__ == "__main__":
    main()
