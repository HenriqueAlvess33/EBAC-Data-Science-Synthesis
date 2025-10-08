import pandas as pd
import numpy as np
import streamlit as st
import requests
import tensorflow as tf
import pytz
import time
from datetime import datetime
from transformers import pipeline
from pydantic import BaseModel
from tensorflow.keras.models import Model


@st.cache_data(ttl=3600)
def load_data(url):
    df = pd.read_csv(url)  # 👈 Download the data
    return df


df = load_data("https://github.com/plotly/datasets/raw/master/uber-rides-data1.csv")
st.dataframe(df)

st.button("Rerun")

st.markdown("---")


@st.cache_data(ttl=3600)
def transform(df):
    df = df.filter(items=["one", "three"])
    df = df.apply(np.sum, axis=0)
    return df


@st.cache_data(ttl=3600)
def transform(df):
    df = df.filter(items=["one", "three"])
    df = df.apply(np.sum, axis=0)
    return df


@st.cache_data(ttl=3600)
def add(arr1, arr2):
    return arr1 + arr2


# connection = database.connect()


@st.cache_data(ttl=3600)
def query():
    return pd.read_sql_query("SELECT * from table", connection)


@st.cache_data(ttl=3600)
def api_call():
    response = requests.get("https://jsonplaceholder.typicode.com/posts/1")
    return response.json()


@st.cache_data
def run_model(inputs):
    return model(inputs)


# Aplicação prática de @st.cache_resource para armazenar em cachê modelos de ML
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")


model = load_model()

query = st.text_input("Your query", value="I love Streamlit! 🎈")
if query:
    result = model(query)[0]
    st.write(result)


# Exemplo de conexão com um banco de dados
# @st.cache_resource
# def init_connection():
#     host = "hh-pgsql-public.ebi.ac.uk"
#     database = "pfmegrnargs"
#     user = "reader"
#     password = "NWDMCE5xdipIjRrp"
#     return psycopg2.connect(host=host, database=database, user=user, password=password)

# conn = init_connection()

# ---

# @st.cache_resource
# def load_model():
#     model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
#     model.eval()
#     return model

# model = load_model()

# Utilização de atributo da classe nos parâmetros da função em cachê, necessário passar nos parâmetros de @st.cache_data() uma maneira de se realizar o hash para a classe MyCustomClass
# class MyCustomClass:
#     def __init__(self, initial_score: int):
#         self.my_score = initial_score


# def hash_func(obj: MyCustomClass) -> int:
#     return obj.my_score

st.markdown("---")

# @st.cache_data(hash_funcs={MyCustomClass: hash_func})
# def multiply_score(obj: MyCustomClass, multiplier: int) -> int:
#     return obj.my_score * multiplier


# initial_score = st.number_input("Enter initial score", value=15)

# score = MyCustomClass(initial_score)
# multiplier = 2

# st.write(multiply_score(score, multiplier))


# Utilizando recurso de cache data de dentro da classe, evitando utilizar a classe em si como parâmetro


class MyCustomClass:
    def __init__(self, initial_score: int):
        self.my_score = initial_score

    @st.cache_data(
        hash_funcs={"__main__.MyCustomClass": lambda x: hash(x.my_score)},
        show_spinner="Fetching data from API...",
    )
    def multiply_score(self, multiplier: int) -> int:
        return self.my_score * multiplier


initial_score = st.number_input("Enter initial score", value=15)

score = MyCustomClass(initial_score)
multiplier = 2

st.write(score.multiply_score(multiplier))

st.markdown("---")


class Person(BaseModel):
    name: str


@st.cache_data(hash_funcs={Person: lambda p: p.name})
def identity(person: Person):
    return person


person = identity(Person(name="Lee"))
st.write(f"The person is {person.name}")

st.markdown("---")


# Caching machine learning models
@st.cache_resource
def load_base_model(option):
    if option == 1:
        return tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
    else:
        return tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet")


@st.cache_resource(hash_funcs={Model: lambda x: x.name})
def load_layers(base_model):
    return [layer.name for layer in base_model.layers]


option = st.radio("Model 1 or 2", [1, 2])

base_model = load_base_model(option)

layers = load_layers(base_model)

st.write(layers)

st.markdown("---")

tz = pytz.timezone("Europe/Berlin")


@st.cache_data(hash_funcs={datetime: lambda x: x.strftime("%a %d %b %Y, %I:%M%p")})
def load_data(dt):
    return dt


now = datetime.now()
st.text(load_data(dt=now))

now_tz = tz.localize(datetime.now())
st.text(load_data(dt=now_tz))

st.markdown("---")


@st.cache_data
def get_data():
    df = pd.DataFrame({"num": [112, 112, 2, 3], "str": ["be", "a", "be", "c"]})
    return df


@st.cache_data(hash_funcs={np.ndarray: str})
def show_data(data):
    time.sleep(2)
    return data


df = get_data()
data = df["str"].unique()

st.dataframe(show_data(data))
st.button("Re-run")

# Exemplo da possibilidade de utilização de comandos streamlit dentro de funções com @st.cache_data
# @st.cache_data
# def get_api_data():
#     data = api.get(...)
#     st.sucess('Fetched data from API!')
#     return data


# Criação de toda uma interface de usuário de dentro da função utilizando comandos streamlit, que mesmo a função estando em cache, eles são executados conforme a leitura do programa
# @st.cache_data
# def show_data():
#     st.header("Data analysis")
#     data = api.get(...)
#     st.success("Fetched data from API!")
#     st.write("Here is a plot of the data:")
#     st.line_chart(data)
#     st.write("And here is the raw data:")
#     st.dataframe(data)

# Inclusão de widgets para input de dados dentro de funções com @st.cache_data
# @st.cache_data(experimental_allow_widgets=True)  # 👈 Set the parameter
# def get_data():
#     num_rows = st.slider("Number of rows to get")  # 👈 Add a slider
#     data = api.get(..., num_rows)
#     return data
