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


class TabelaDeResultados:
    def __init__(self, seed=42):
        self.seed = seed
        self.n_alunos = 100

        # Criar todos os resultados já no início, sempre iguais
        rng = np.random.default_rng(self.seed)
        self._matematica = pd.DataFrame(
            rng.integers(0, 24, size=(self.n_alunos, 1)), columns=["Qt_acertos"]
        )
        self._portugues = pd.DataFrame(
            rng.integers(0, 18, size=(self.n_alunos, 1)), columns=["Qt_acertos"]
        )
        self._geografia = pd.DataFrame(
            rng.integers(0, 8, size=(self.n_alunos, 1)), columns=["Qt_acertos"]
        )
        self._ingles = pd.DataFrame(
            rng.integers(0, 8, size=(self.n_alunos, 1)), columns=["Qt_acertos"]
        )
        self._historia = pd.DataFrame(
            rng.integers(0, 8, size=(self.n_alunos, 1)), columns=["Qt_acertos"]
        )
        self._fisica = pd.DataFrame(
            rng.integers(0, 12, size=(self.n_alunos, 1)), columns=["Qt_acertos"]
        )
        self._quimica = pd.DataFrame(
            rng.integers(0, 12, size=(self.n_alunos, 1)), columns=["Qt_acertos"]
        )

    # Matemática
    def resultados_matematica(self):
        return self._matematica.copy()

    def percentual_de_acertos_matematica(self):
        df = self.resultados_matematica()
        df["Percentual_de_acertos"] = (df["Qt_acertos"] / 24) * 100
        return df

    # Português
    def resultados_portugues(self):
        return self._portugues.copy()

    def percentual_de_acertos_portugues(self):
        df = self.resultados_portugues()
        df["Percentual_de_acertos"] = (df["Qt_acertos"] / 18) * 100
        return df

    # Geografia
    def resultados_geografia(self):
        return self._geografia.copy()

    def percentual_de_acertos_geografia(self):
        df = self.resultados_geografia()
        df["Percentual_de_acertos"] = (df["Qt_acertos"] / 8) * 100
        return df

    # Inglês
    def resultados_ingles(self):
        return self._ingles.copy()

    def percentual_de_acertos_ingles(self):
        df = self.resultados_ingles()
        df["Percentual_de_acertos"] = (df["Qt_acertos"] / 8) * 100
        return df

    # História
    def resultados_historia(self):
        return self._historia.copy()

    def percentual_de_acertos_historia(self):
        df = self.resultados_historia()
        df["Percentual_de_acertos"] = (df["Qt_acertos"] / 8) * 100
        return df

    # Física
    def resultados_fisica(self):
        return self._fisica.copy()

    def percentual_de_acertos_fisica(self):
        df = self.resultados_fisica()
        df["Percentual_de_acertos"] = (df["Qt_acertos"] / 12) * 100
        return df

    # Química
    def resultados_quimica(self):
        return self._quimica.copy()

    def percentual_de_acertos_quimica(self):
        df = self.resultados_quimica()
        df["Percentual_de_acertos"] = (df["Qt_acertos"] / 12) * 100
        return df

    # Todos
    def todos_os_resultados(self):
        df = pd.DataFrame(
            {
                "Matemática": self._matematica["Qt_acertos"],
                "Português": self._portugues["Qt_acertos"],
                "Geografia": self._geografia["Qt_acertos"],
                "Inglês": self._ingles["Qt_acertos"],
                "História": self._historia["Qt_acertos"],
                "Física": self._fisica["Qt_acertos"],
                "Química": self._quimica["Qt_acertos"],
            }
        )

        return df

    def dataframe_porcentagem_de_acerto(self):
        df = self.todos_os_resultados()
        df["Total_de_acertos"] = df.sum(axis=1)
        perguntas_por_materia = {
            "Matemática": 24,
            "Português": 18,
            "Geografia": 8,
            "Inglês": 8,
            "História": 8,
            "Física": 12,
            "Química": 12,
        }
        total_de_perguntas = sum(perguntas_por_materia.values())
        df["Porcentagem de acertos"] = round(
            (df["Total_de_acertos"] / total_de_perguntas) * 100, 3
        )

        return df


@st.cache_data
def get_tabela_resultados():
    return TabelaDeResultados(seed=42)


def main():

    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">

        <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 3em;'>
            <strong>Portal de performance do Vestibular SYT 2025</strong>
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # Criar visualizacao de dataframes, por matéria e no geral
    tab_visualizacao_dataframes, tab_alunos_aprovados = st.tabs(
        [
            "Visualização geral da performance dos alunos por matéria",
            "Aprovados e Reprovados",
        ]
    )

    with tab_visualizacao_dataframes:
        col1, col2, col3 = st.columns([1, 1, 1])
        materia_selecionada = col1.selectbox(
            "Selecione a matéria que deseja visualizar os resultados individualmente",
            [
                "Matemática",
                "Português",
                "Geografia",
                "Inglês",
                "História",
                "Física",
                "Química",
            ],
        )

        botao_confirmacao = col1.button("Selecionar matéria")

        if "registro_da_matéria" not in st.session_state:
            st.session_state.registro_da_matéria = None

        if (materia_selecionada == "Matemática") and (botao_confirmacao):
            col1.success("Número de acertos nas questões de matemática")
            col1.write(f"A prova de matemática possui 24 questões")
            st.session_state.registro_da_matéria = (
                get_tabela_resultados().percentual_de_acertos_matematica()
            )
        elif (materia_selecionada == "Português") and (botao_confirmacao):
            col1.success("Número de acertos nas questões de português")
            col1.write(f"A prova de português possui 18 questões")
            st.session_state.registro_da_matéria = (
                get_tabela_resultados().percentual_de_acertos_portugues()
            )
        elif (materia_selecionada == "Geografia") and (botao_confirmacao):
            col1.success("Número de acertos nas questões de geografia")
            col1.write(f"A prova de georgrafia possui 8 questões")
            st.session_state.registro_da_matéria = (
                get_tabela_resultados().percentual_de_acertos_geografia()
            )
        elif (materia_selecionada == "Inglês") and (botao_confirmacao):
            col1.success("Número de acertos nas questões de inglês")
            col1.write(f"A prova de inglês possui 8 questões")
            st.session_state.registro_da_matéria = (
                get_tabela_resultados().percentual_de_acertos_ingles()
            )
        elif (materia_selecionada == "História") and (botao_confirmacao):
            col1.success("Número de acertos nas questões de história")
            col1.write(f"A prova de história possui 8 questões")
            st.session_state.registro_da_matéria = (
                get_tabela_resultados().percentual_de_acertos_historia()
            )
        elif (materia_selecionada == "Física") and (botao_confirmacao):
            col1.success("Número de acertos nas questões de física")
            col1.write(f"A prova de física possui 12 questões")
            st.session_state.registro_da_matéria = (
                get_tabela_resultados().percentual_de_acertos_fisica()
            )
        elif (materia_selecionada == "Química") and (botao_confirmacao):
            col1.success("Número de acertos nas questões de química")
            col1.write(f"A prova de química possui 12 questões")
            st.session_state.registro_da_matéria = (
                get_tabela_resultados().percentual_de_acertos_quimica()
            )

        # Só mostra se a variável tiver sido definida
        if st.session_state.registro_da_matéria is not None:
            col1.dataframe(st.session_state.registro_da_matéria)

        if st.checkbox("Demonstrar dataframe com todas as matérias"):
            resultados_acertos = st.dataframe(
                get_tabela_resultados().dataframe_porcentagem_de_acerto()
            )
    with tab_alunos_aprovados:
        col1, col2 = st.columns([1, 1])
        dataframe_basal = get_tabela_resultados().dataframe_porcentagem_de_acerto()
        dataframe_aprovados = dataframe_basal.loc[
            dataframe_basal["Porcentagem de acertos"] >= 45
        ]
        dataframe_reprovados = dataframe_basal.loc[
            dataframe_basal["Porcentagem de acertos"] < 45
        ]
        if col1.button("Lista de Alunos Aprovados"):
            st.success(
                f"A quantidade de alunos aprovados foi de: {dataframe_aprovados.shape[0]}"
            )
            st.dataframe(dataframe_aprovados)
        if col2.button("Lista de Alunos Reprovados"):
            st.error(
                f"A quantidade de alunos reprovados foi de: {dataframe_reprovados.shape[0]}"
            )
            st.dataframe(dataframe_reprovados)


if __name__ == "__main__":
    main()
