import streamlit as st
import pandas as pd
import numpy as np
import time
import pytz
import time
from datetime import datetime
from pydantic import BaseModel
from transformers import pipeline
import tensorflow as tf
from tensorflow.keras.models import Model

st.set_page_config(layout="wide")

st.markdown(
    """
    <div style="text-align: center; font-size: 48px; font-weight: bold; color: #2c3e50; margin-top: 20px;">
        Uber pickups in NYC
    </div>
    """,
    unsafe_allow_html=True,
)

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


data_load_state = st.text("Loading data...")
data = load_data(10000)
data_load_state.text("Loading data...done!")

if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.dataframe(data)

st.markdown(
    """
    <div style="text-align: left; font-size: 30px;  color: #2c3e50; margin-top: 20px; margin-bottom: 20px;">
        <em>Number of pickups per hour</em>
    </div>
    """,
    unsafe_allow_html=True,
)
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
st.bar_chart(hist_values)

hour_to_filter = st.slider("hour", 0, 23, 17)  # min: 0h, max: 23h, default: 17h
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f"Map of all pickups at {hour_to_filter}:00")
st.map(filtered_data)


def lbs_to_kg():
    st.session_state.kg = st.session_state.lbs / 2.2046


def kg_to_lbs():
    st.session_state.lbs = st.session_state.kg * 2.2046


def kilometers_to_miles():
    st.session_state.mi = st.session_state.km / 1.6093


def miles_to_kilometers():
    st.session_state.km = st.session_state.mi * 1.6093


col1, buff, col2 = st.columns([2, 1, 2])

with col1:
    pounds = st.sidebar.number_input("Pounds:", key="lbs", on_change=lbs_to_kg)
    kilogram = st.sidebar.number_input("Kilograms:", key="kg", on_change=kg_to_lbs)


with col2:

    miles = st.sidebar.number_input("Miles", key="mi", on_change=miles_to_kilometers)
    kilometers = st.sidebar.number_input(
        "Kilometers", key="km", on_change=kilometers_to_miles
    )
st.markdown("---")
if (
    "counter" not in st.session_state
):  # Condição atendida na primeira execução, onde é verificada a existência da chave "counter" e criada caso não exista
    st.session_state.counter = 0

st.session_state.counter += 1  # Acrécsimo ao valor salvo para cada execução de código

st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")

st.markdown("---")
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("Choose a datapoint color")
color = st.color_picker("Color", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y", color=color)

st.markdown("---")

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.line_chart(chart_data)

st.markdown("---")

left_column, right_column = st.columns(2)

left_column.button("Press me!")


with right_column:
    chosen = st.radio(
        "Sorting hat", ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin")
    )
st.write(f"You are {chosen} house!")

st.markdown("---")

"Starting a long computation"

latest_iteration = st.empty()
bar = st.progress(0)

for i in range(0, 100):
    latest_iteration.text(f"Iteration {i+1}")
    bar.progress(i + 1)
    time.sleep(0.1)

"...and now we're done!"

st.markdown("---")


@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")


model = load_model()

query = st.text_input("Your query", value="I love Streamlit! 🎈")
if query:
    result = model(query)[0]
    st.write(result)

st.markdown("---")


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

st.markdown("---")

if "a_counter" not in st.session_state:
    st.session_state["a_counter"] = 0

if "boolean" not in st.session_state:
    st.session_state.boolean = False

if "key" not in st.session_state:
    st.session_state.key = "value"

st.write("a counter is:", st.session_state["a_counter"])
st.write("boolean is:", st.session_state.boolean)

button = st.button("Update state")
"before pressing button", st.session_state

if button:
    st.session_state.a_counter += 1
    st.session_state.boolean = not st.session_state.boolean
    "after pressing button", st.session_state


st.markdown("---")

number = st.slider("A number", 1, 10, key="slider")

col1, buff, col2 = st.columns([1, 0.5, 3])

option_names = ["a", "b", "c"]

next = st.button("Next option")

if next:
    if st.session_state["radio_option"] == "a":
        st.session_state.radio_option = "b"

    elif st.session_state["radio_option"] == "b":
        st.session_state.radio_option = "c"

    elif st.session_state["radio_option"] == "c":
        st.session_state.radio_option = "a"

option = col1.radio("Pick an option", option_names, key="radio_option")
"Session state is: ", st.session_state

if option == "a":
    col2.write("You picked 'a' :smile:")
elif option == "b":
    col2.write("You picked 'b' :heart:")
else:
    col2.write("You picked 'c' :rocket:")

st.markdown("---")

st.session_state.key = "value2"

st.write(st.session_state)


st.text_input("Your name", key="name")


def form_callback():
    st.write(st.session_state.my_slider)
    st.write(st.session_state.my_checkbox)


with st.form(key="my_form"):
    slider_input = st.slider("My slider", 0, 10, 5, key="my_slider")
    checkbox_input = st.checkbox("Yes or no", key="my_checkbox")
    submit_button = st.form_submit_button(label="Submit", on_click=form_callback)

add_slider = st.sidebar.slider("Select a range of values", 0.0, 100.0, (25.0, 75.0))
st.markdown("---")

add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?", ("Email", "Home Phone", "Mobile Phone")
)
