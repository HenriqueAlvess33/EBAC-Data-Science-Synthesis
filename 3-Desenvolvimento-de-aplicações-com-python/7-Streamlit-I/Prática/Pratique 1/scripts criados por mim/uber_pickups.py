import streamlit as st
import pandas as pd
import numpy as np

st.title("Uber pickups in NYC")

DATE_COLUMN = "date/time"
DATA_URL = (
    "https://s3-us-west-2.amazonaws.com/"
    "streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)


@st.cache_data  # Ferramenta importante para que o dataframe não seja recarregado em todo salvamento, guardando as informações em cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


# Informa o leitor que os dados estão sendo carregados
data_load_state = st.text("Loading data...")

# Carrega 10.000 linhas requeridas no parâmetro
data = load_data(10000)

# Notifica o leitor que o dado foi carregado corretamente
# data_load_state.text("Loading data...done!")
data_load_state.text(
    "Done! (using st.cache_data)"
)  # A troca de frase em comparação a linha superior é instantânea por conta do cache


# st.subheader("Raw data")
# st.dataframe(data)

# Criar ferramenta de checkbox para exibir o dataframe sem modificações
if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(data)

# Desenhando um Histograma
st.subheader("Number of pickups per hour")
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
st.bar_chart(hist_values)

# Desenhando a distribuição de pickups pelo mapa de Nova York
# st.subheader("Map of all pickups")
# st.map(data)

# Filtrando todo inicio de corrida para o hotário mais movimentado, que são as 17:00 de acordo com o histograma
# hour_to_filter = 17
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# st.subheader(f"Map of all pickups at {hour_to_filter}:00")

# st.map(filtered_data)

# Adição de uma widget de slider para que o usuário possa realizar a filtragemm dos horários
hour_to_filter = st.slider("hour", 0, 23, 17)  # min: 0h, max: 23h, default: 17h
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f"Map of all pickups at {hour_to_filter}:00")
st.map(filtered_data)
