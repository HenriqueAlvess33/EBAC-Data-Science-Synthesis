import streamlit as st
import pandas as pd
import numpy as np

# Configura o título da aplicação
st.set_page_config(
    page_title="Entendimento sobre pd.Series",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="varig_icon.png",
)


@st.cache_data
def get_series_data():
    anomalia_termica = [-0.08, -0.27, 0.12, -0.03, 0.26, 0.40, 1.02]
    decadas_registradas = [1900, 1920, 1940, 1960, 1980, 2000, 2020]
    return pd.Series(anomalia_termica, index=decadas_registradas)


def main():

    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">

    <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 3em;'>
        <strong>Atividades Introdutórias ao Python</strong>
    </h1>
    """,
        unsafe_allow_html=True,
    )

    relacao_decadas_temperatura_mundial = (get_series_data(),)

    explanacao_da_tarefa_tab, simulacao_de_problema_tab = st.tabs(
        ["Explicação da tarefa e de seus objetivos", "Trabalho realizado"]
    )
    with explanacao_da_tarefa_tab:
        st.markdown(
            """# Tarefa 01

        - Leia os enunciados com atenção
        - Saiba que pode haver mais de uma resposta correta
        - Insira novas células de código sempre que achar necessário
        - Em caso de dúvidas, procure os tutores
        - Divirta-se :)

        #### 1)  crie uma série do pandas a partir de uma lista com os dados abaixo:

        Em um estudo sobre alteração na tempreatura global, A NASA disponibiliza dados de diferenças de de temperatura média da superfície terrestre relativos às médias de temperatura entre 1951 e 1980. Os dados originais podem ser vistos no site da NASA/GISS, e estão dispostos a cada década na tabela abaixo.

        |ano|anomalia termica|
        |:-:|:----:|
        | 1900 | -0.08 |
        | 1920 | -0.27 |
        | 1940 | 0.12 |
        | 1960 | -0.03 |
        | 1980 | 0.26 |
        | 2000 | 0.40 |
        | 2020 | 1.02 |

        Crie uma séries do Pandas a partir de uma lista com esses dados.)
        """,
            unsafe_allow_html=True,
        )
    with simulacao_de_problema_tab:
        aba1, aba2 = st.tabs(
            [
                "Transformação de listas em series",
                "Conversão de arrays em dataframe e criação de novas variáveis",
            ]
        )

        with aba1:

            # Seu código aqui
            st.subheader("Demonstração de listas a serem convertidas em Series")
            anomalia_termica = [-0.08, -0.27, 0.12, -0.03, 0.26, 0.40, 1.02]
            decadas_registradas = [1900, 1920, 1940, 1960, 1980, 2000, 2020]

            st.write(f"Lista de variações na temperatura: {anomalia_termica}")
            st.write(f"Lista dos anos documentados: {decadas_registradas}")

            relacao_decadas_temperatura_mundial = get_series_data()
            st.subheader("Série Pandas resultante:")
            st.dataframe(relacao_decadas_temperatura_mundial)

        with aba2:
            st.header(
                "Capacidade do Pandas organizar arrays com números aleatórios em um dataframe coeso"
            )
            st.write(
                "Com pandas temos a capacidade de organizar qualquer matriz de dados de uma maneira que facilite a leitura e interpretação das informações"
            )
            with st.expander(
                "Conjunto de dados resultante de um array com números aleatórios",
                expanded=False,
            ):
                np.random.seed(42)
                arr = np.random.normal(100, 10, (20, 3))
                arr = (
                    pd.DataFrame(arr, columns=["X1", "X2", "X3"])
                    .assign(media=lambda df: df.mean(axis=1))
                    .assign(log_med=lambda df: np.log(df["media"]))
                )
                st.write("")

                st.dataframe(arr)


if __name__ == "__main__":
    main()
