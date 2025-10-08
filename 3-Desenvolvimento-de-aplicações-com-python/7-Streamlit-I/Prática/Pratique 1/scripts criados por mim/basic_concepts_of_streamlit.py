import streamlit as st
import pandas as pd
import numpy as np
import time

# Criação e visualização de um dataframe
df = pd.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})
df
st.markdown("---")

# Escrita e visualização de um dataframe
st.write("Here's our first attempt at using data to create table:")
st.write(
    pd.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})
)

# Criação de um dataframe e exibição pelo método st.dataframe()
dataframe = np.random.randn(10, 20)
st.dataframe(dataframe)
st.markdown("---")

# Criação de Dataframe interativo que esteja em evidência os maiores valores de cada coluna
dataframe = pd.DataFrame(
    np.random.randn(10, 20), columns=("col %d" % i for i in range(20))
)
st.dataframe(dataframe.style.highlight_max(axis=0))
st.markdown("---")

# Criação de tabela estática para demonstrar todos os dados do dataframe
dataframe = pd.DataFrame(
    np.random.randn(10, 20), columns=("col %d" % i for i in range(20))
)
st.table(dataframe)
st.markdown("---")

# Criação de gráfico de linha com os dados da tabela
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.line_chart(chart_data)
st.markdown("---")

# Plotagem de mapa
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76 - 122.4], columns=["lat", "lon"]
)
st.map(map_data)
st.markdown("---")

# Construção de uma widget de rolagem para determinar o valor de x
x = st.slider("x")
st.write(x, "squared is", x * x)
st.markdown("---")

# Cria uma checkbox com o titulo "Show dataframe" e exibe o dataframe quando preenchida
if st.checkbox("Show dataframe"):
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    chart_data
st.markdown("---")

# Ferramenta para criar uma caixa de seleção com base nas variáveis de uma coluna informada do dataframe
df = pd.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})

option = st.selectbox("Which number do you like best ?", df["first column"])

"You selected: ", option
st.markdown("---")


# O atributo .sidebar permite que coloquemos os recursos em uma barra lateral
# É criado uma caixa de seleção com os valores de string apresentado no parênteses interno dos parâmetros
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?", ("Email", "Home Phone", "Mobile Phone")
)
# # # Adiciona uma widget de rolagem mas com os valores padrões setados para 25 e 75, desta maneira temos uma barra com dois pontos, um fixado em cada valor pré-setado
add_slider = st.sidebar.slider("Select a range of values", 0.0, 100.0, (25.0, 75.0))
st.markdown("---")

# Podemos dividir os recursos da página em colunas, desta maneira setorizando a página para dividir as visualizações
left_column, right_column = st.columns(2)

left_column.button("Press me!")


with right_column:
    chosen = st.radio(
        "Sorting hat", ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin")
    )
st.write(f"You are {chosen} house!")
st.markdown("---")

# Função "Show progress", por este meio podemos criar uma visualização em tempo real do progresso de leitura de alfum programa
# Iremos utilzar a biblioteca nativa de python "time" para simular estas operações duradouras
# Imprime a primeira string de inicio na tela e após a contage e conclusão da barra de progresso a segunda string é exibida

"Starting a long computation"

latest_iteration = st.empty()
bar = st.progress(0)

for i in range(0, 100):
    latest_iteration.text(f"Iteration {i+1}")
    bar.progress(i + 1)
    time.sleep(0.1)

"...and now we're done!"
st.markdown("---")
