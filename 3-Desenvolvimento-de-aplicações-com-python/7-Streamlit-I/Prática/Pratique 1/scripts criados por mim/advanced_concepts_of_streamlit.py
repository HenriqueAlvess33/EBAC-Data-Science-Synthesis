import streamlit as st
import pandas as pd
import numpy as np

if (
    "counter" not in st.session_state
):  # Condição atendida na primeira execução, onde é verificada a existência da chave "counter" e criada caso não exista
    st.session_state.counter = 0

st.session_state.counter += 1  # Acrécsimo ao valor salvo para cada execução de código

st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")


# Criação de uma base de dados aleatória que é salva após a primeira execução para se manter igual
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("Choose a datapoint color")
color = st.color_picker("Color", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y", color=color)

# Exemplo de como se realizar uma conexão com bases de dados no SQL
# conn = st.connection("my_database")
# df = conn.query("select * from my table")
# st.dataframe(df)
