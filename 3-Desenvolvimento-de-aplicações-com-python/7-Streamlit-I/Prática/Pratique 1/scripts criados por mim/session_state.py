import streamlit as st

st.title("Session state basics")

"Session state is: ", st.session_state

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


def lbs_to_kg():
    st.session_state.kg = st.session_state.lbs / 2.2046


def kg_to_lbs():
    st.session_state.lbs = st.session_state.kg * 2.2046


col1, buff, col2 = st.columns([2, 1, 2])

with col1:
    pounds = st.number_input("Pounds:", key="lbs", on_change=lbs_to_kg)

with col2:
    kilogram = st.number_input("Kilograms:", key="kg", on_change=kg_to_lbs)


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
