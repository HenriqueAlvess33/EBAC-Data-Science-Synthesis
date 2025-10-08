import numpy as np
import pandas as pd
import streamlit as st


# st.write("Bem vinda")

# # dataframe = np.random.randn(10, 20)
# # st.dataframe(dataframe)

# # dataframe = pd.DataFrame(
# #     np.random.randn(10, 20), columns=("col %d" % i for i in range(20))
# # )


# # # Cria uma outra visualização do dataframe em formato de tabela
# # dataframe = np.random.randn(10, 20)
# # st.dataframe(dataframe)

# # dataframe = pd.DataFrame(
# #     np.random.randn(10, 20), columns=("col %d" % i for i in range(20))
# # )
# # st.table(dataframe)

# x = st.slider("x")
# st.write(x, "squared is", x * x)

"""
# My first app
Here's our first attempt at using data to create a table:
"""

df = pd.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})

df
