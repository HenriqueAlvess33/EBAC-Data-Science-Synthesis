# Bibliotecas padrão
import os

import io

# Desenvolvimento de aplicação online
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin

# Manipulação de dados
import pandas as pd
import numpy as np

from IPython.display import (
    display,
    HTML,
)  # Exibição de tabelas e HTML no Jupyter Notebook

# Visualização de dados
import seaborn as sns
import matplotlib.pyplot as plt

# Modelagem estatística
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


def main():
    # Configura o título da aplicação
    st.set_page_config(
        page_title="Previsão de renda dos clientes",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="varig_icon.png",
    )
    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">

    <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 3em;'>
        <strong>Previsão de renda dos clientes</strong>
    </h1>
    """,
        unsafe_allow_html=True,
    )


# def modelando_arvore_regressao(_X_treino, _y_treino):

#     # 1. Extração dos valores possíveis para o parâmetro de poda ccp_alpha
#     # O método cost_complexity_pruning_path retorna os valores de ccp_alpha que podem ser usados para podar a árvore.
#     path = DecisionTreeRegressor(random_state=42).cost_complexity_pruning_path(
#         X_treino, y_treino
#     )
#     ccp_alphas = path.ccp_alphas

#     # Para evitar que o grid fique muito pesado, limitamos os valores de ccp_alpha para valores não-negativos
#     ccp_alphas = [a for a in ccp_alphas if a >= 0][:30]

#     # 2. Definição da grade de hiperparâmetros para busca
#     # max_depth: profundidade máxima da árvore (quanto maior, mais complexa)
#     # min_samples_leaf: número mínimo de amostras em uma folha (evita folhas com poucos dados)
#     # ccp_alpha: parâmetro de custo-complexidade para poda (quanto maior, mais agressiva a poda)
#     params = {
#         "max_depth": [3, 4, 5, 6],
#         "min_samples_leaf": [2, 3, 4, 5],
#         "ccp_alpha": ccp_alphas,
#     }

#     # 3. Execução do GridSearchCV
#     # GridSearchCV realiza busca exaustiva combinando todos os valores possíveis dos hiperparâmetros definidos.
#     # cv=5: utiliza validação cruzada com 5 folds para avaliar cada combinação.
#     # scoring="r2": métrica utilizada é o R² (coeficiente de determinação).
#     # n_jobs=-1: utiliza todos os núcleos disponíveis para acelerar o processo.
#     grid = GridSearchCV(
#         estimator=DecisionTreeRegressor(random_state=42),
#         param_grid=params,
#         cv=5,  # 5-fold cross-validation
#         scoring="r2",
#         n_jobs=-1,
#     )

#     # Ajuste do modelo com os dados de treino
#     grid.fit(_X_treino, _y_treino)

#     return grid


def modelando_arvore_regressao(_X_treino, _y_treino):

    # Verificar se há strings nos dados de entrada
    string_cols = _X_treino.select_dtypes(include=["object"]).columns
    if len(string_cols) > 0:
        st.error(f"❌ STRINGS ENCONTRADAS na entrada da função!: {list(string_cols)}")
        for col in string_cols:
            st.error(f"Coluna {col}: {_X_treino[col].unique()}")
        # Forçar conversão para numérico
        _X_treino = _X_treino.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 1. Extração dos valores possíveis para o parâmetro de poda ccp_alpha
    try:
        path = DecisionTreeRegressor(random_state=42).cost_complexity_pruning_path(
            _X_treino, _y_treino
        )
        ccp_alphas = path.ccp_alphas
    except Exception as e:
        st.error(f"❌ Erro no cost_complexity_pruning_path: {e}")
        # Fallback: usar valores default para ccp_alpha
        ccp_alphas = [0.0, 0.001, 0.01, 0.1]
        st.warning(f"Usando ccp_alphas default: {ccp_alphas}")

    # Para evitar que o grid fique muito pesado, limitamos os valores de ccp_alpha para valores não-negativos
    ccp_alphas = [a for a in ccp_alphas if a >= 0][:30]

    # 2. Definição da grade de hiperparâmetros para busca
    params = {
        "max_depth": [3, 4, 5, 6],
        "min_samples_leaf": [2, 3, 4, 5],
        "ccp_alpha": ccp_alphas,
    }

    # 3. Execução do GridSearchCV
    grid = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=params,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    # Ajuste do modelo com os dados de treino
    try:
        grid.fit(_X_treino, _y_treino)
        st.success("✅ GridSearchCV concluído com sucesso!")
    except Exception as e:
        st.error(f"❌ Erro no GridSearchCV: {e}")
        # Fallback: modelo simples
        st.warning("Usando DecisionTreeRegressor padrão como fallback")
        model = DecisionTreeRegressor(random_state=42, max_depth=5)
        model.fit(_X_treino, _y_treino)
        return model

    return grid


def stepwise_selection(
    X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True
):
    """Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            print(included + [new_column])
            model = sm.OLS(
                y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))
            ).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()  # retorna o nome da coluna
            included.append(best_feature)
            changed = True
            if verbose:
                print("Add  {:30} with p-value {:.6}".format(best_feature, best_pval))

        # backward step
        print(included)
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()  # retorna o nome da coluna
            included.remove(worst_feature)
            if verbose:
                print("Drop {:30} with p-value {:.6}".format(worst_feature, worst_pval))
        if not changed:
            break
    return included


def load_data(file_data):
    if file_data.name.endswith(".csv"):
        # Detecta automaticamente , ou ;
        return pd.read_csv(file_data, sep=None, engine="python")
    elif file_data.name.endswith(".xlsx"):
        return pd.read_excel(file_data)
    else:
        raise ValueError("Formato de arquivo não suportado")


def proporcao_de_categorias(dataframe, var):

    contagem = pd.crosstab(index=dataframe[var], columns="Frequência")
    contagem.sort_values(by="Frequência", ascending=False, inplace=True)
    contagem["pct_freq"] = (contagem["Frequência"] / contagem["Frequência"].sum()) * 100
    contagem.sort_values(by="Frequência", ascending=False, inplace=True)

    # Plotagem de gráfico
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=contagem.index, y=contagem["pct_freq"], ax=ax)
    ax.set_ylabel("Frequência em percentual (%)")

    ax.set_ylabel("Frequencia em percentual %")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=800)

    return contagem


def descritiva_bivariada(categorica, numerica, df, setup):
    if setup == "boxplot":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df, x=categorica, y=numerica, ax=ax)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, width=800)
    elif setup == "violinplot":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(data=df, x=categorica, y=numerica, ax=ax)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, width=800)
    elif setup == "barplot":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=df, x=categorica, y=numerica, ax=ax)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, width=800)


def diagnostico_de_duplicidade(df, string_de_id):
    ids_duplicados = [
        id_
        for id_, quantidade in df[string_de_id].value_counts().items()
        if quantidade > 1
    ]

    df_duplicados = df[df[string_de_id].isin(ids_duplicados)]

    return ids_duplicados, df_duplicados


def preprocess_input(input_df, training_columns):
    """
    Pré-processa novos dados da mesma forma que os dados de treino
    """
    # Separar colunas como foi feito no treino
    colunas_categoricas = ["tipo_renda", "educacao", "estado_civil", "tipo_residencia"]
    colunas_numericas = [
        col for col in input_df.columns if col not in colunas_categoricas
    ]

    # Processar numéricas
    input_numerico = input_df[colunas_numericas].copy()

    # Processar categóricas (get_dummies)
    input_categorico_dummies = pd.get_dummies(
        input_df[colunas_categoricas], drop_first=True
    ).astype(int)

    # Juntar tudo
    input_processed = pd.concat([input_numerico, input_categorico_dummies], axis=1)

    # Garantir que tem todas as colunas do treino
    for col in training_columns:
        if col not in input_processed.columns:
            input_processed[col] = 0

    # Reordenar colunas exatamente como no treino
    input_processed = input_processed[training_columns]

    return input_processed


class OutlierTreatment(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        _df = X.copy()
        _df_scaled = StandardScaler().fit_transform(_df)
        iso = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        )
        pred = iso.fit_predict(_df_scaled)
        _df["outlier_iforest"] = pred == -1
        return _df


main()  # Executa a função principal da aplicação (provavelmente inicializa a interface Streamlit)

# Permite ao usuário fazer upload de arquivo CSV ou XLSX pela barra lateral do Streamlit
upload_file = st.sidebar.file_uploader("previsão_de_renda", type=["csv", "xlsx"])

if upload_file is not None:
    # Carrega o arquivo no session_state apenas uma vez para evitar recarregamentos desnecessários
    if "df_completo" not in st.session_state:
        st.session_state.df_completo = load_data(upload_file)
        # Remove colunas indesejadas se existirem no DataFrame
        if (
            "Unnamed: 0" in st.session_state.df_completo.columns
            and "sexo" in st.session_state.df_completo.columns
        ):
            st.session_state.df_completo = st.session_state.df_completo.drop(
                columns=["Unnamed: 0", "sexo"]
            )

    # Diagnóstico de duplicidade baseado na coluna 'id_cliente'
    _, df_duplicatas = diagnostico_de_duplicidade(
        st.session_state.df_completo, "id_cliente"
    )

    # Cria uma máscara para identificar linhas onde 'tempo_emprego' é nulo e 'tipo_renda' é 'Pensionista'
    mask = (st.session_state.df_completo["tempo_emprego"].isna()) & (
        st.session_state.df_completo["tipo_renda"] == "Pensionista"
    )

    flag_outlier = OutlierTreatment()  # Instancia o tratamento de outliers

    # Aplica o tratamento de outliers após transformar variáveis categóricas em dummies
    diagnostico_outlier = flag_outlier.transform(
        pd.get_dummies(
            st.session_state.df_completo.drop(columns=["data_ref"]),
            drop_first=True,
        )
    )
    # Adiciona a coluna 'outlier_iforest' ao DataFrame principal
    st.session_state.df_completo["outlier_iforest"] = diagnostico_outlier[
        "outlier_iforest"
    ]
    if "arvore_reg" not in st.session_state:
        st.session_state.arvore_reg = None

    if "X_treino_processed" not in st.session_state:
        st.session_state.X_treino_processed = None

    if "y_treino" not in st.session_state:
        st.session_state.y_treino = None

    # Cria uma cópia do DataFrame para manipulação local
    if "df_completo" in st.session_state:
        df = st.session_state.df_completo.copy()

    if "df_visualizacao" not in st.session_state:
        st.session_state.df_visualizacao = st.session_state.df_completo.copy()

    # Garante que a coluna 'Ano' existe, extraindo o ano da coluna 'data_ref'
    if "data_ref" in df.columns and "Ano" not in df.columns:
        df["Ano"] = pd.to_datetime(df["data_ref"]).dt.year

    # Filtra o DataFrame para remover outliers, se a coluna existir
    if "outlier_iforest" in df.columns:
        st.session_state.df_completo_sem_outliers = df.query("outlier_iforest == False")
    else:
        st.warning("A coluna 'outlier_iforest' não foi encontrada no dataset.")

    # Salva DataFrames com e sem outliers no session_state para uso posterior
    if "df_com_outliers" not in st.session_state:
        st.session_state.df_com_outliers = st.session_state.df_completo.query(
            "outlier_iforest == True"
        )

    if "df_sem_outliers" not in st.session_state:
        st.session_state.df_sem_outliers = st.session_state.df_completo.query(
            "outlier_iforest == False"
        )
    if "var_info" not in st.session_state:
        st.session_state.var_info = None

    # Calcula a tabela de valores ausentes por coluna
    st.session_state.tabela_de_ausentes = st.session_state.df_completo.isna().sum()

    # Inicializa a tabela de ausentes como None se não existir
    if "tabela_de_ausentes" not in st.session_state:
        st.session_state["tabela_de_ausentes"] = None

    if "modelo_reg" not in st.session_state:
        st.session_state.modelo_reg = None

    # Cria variável booleana para indicar se o cliente possui dependentes
    st.session_state.df_completo["possui_dependentes"] = np.where(
        st.session_state.df_completo["qtd_filhos"] > 0, 1, 0
    )

    # Cria variável para identificar clientes jovens (idade até 25 anos)
    st.session_state.df_completo["idade_jovem"] = np.where(
        st.session_state.df_completo["idade"] <= 25, 1, 0
    )

    # Calcula índice de instabilidade de carreira (idade dividida pelo tempo de emprego + 1)
    st.session_state.df_completo["idade_x_tempo_emprego"] = (
        st.session_state.df_completo["idade"]
        / (st.session_state.df_completo["tempo_emprego"] + 1)
    )

    # Separa os dados de treino (ano 2015) e teste (ano 2016) sem outliers
    df_treino = st.session_state.df_completo_sem_outliers.loc[
        st.session_state.df_completo_sem_outliers["Ano"] == 2015
    ]
    X_treino = df_treino.drop(columns=["renda", "data_ref"])
    y_treino = df_treino["renda"]
    df_teste_oot = st.session_state.df_completo_sem_outliers.loc[
        st.session_state.df_completo_sem_outliers["Ano"] == 2016
    ]
    X_teste = df_teste_oot.drop(columns=["renda", "data_ref"])
    y_teste = df_teste_oot["renda"]

    if "feature_selection_button" not in st.session_state:
        st.session_state["feature_selection_button"] = False

    if st.session_state.feature_selection_button:
        result = stepwise_selection(
            X_treino.select_dtypes(include=["int64", "float64"]), y_treino
        )
    if "selecao_var_explicativa" not in st.session_state:
        st.session_state.selecao_var_explicativa = None

    if "generate_model_button" not in st.session_state:
        st.session_state.generate_model_button = None

    if st.session_state.selecao_var_explicativa is not None:
        if st.session_state.generate_model_button:
            formula = "+".join(st.session_state.selecao_var_explicativa)
            st.session_state.modelo_reg = smf.ols(
                f"renda ~ {formula}", data=df_treino
            ).fit()

    main_tab_one, main_tab_two = st.tabs(
        ["Análise e tratamento do conjunto de dados", "Desenvolvimento e uso do modelo"]
    )
    if "dados_duplicados" not in st.session_state:
        st.session_state.dados_duplicados = False

    if "dados_tratados" not in st.session_state:
        st.session_state.dados_tratados = False


with main_tab_one:
    # PRIMEIRO expander - Visualização de dataframes
    with st.expander("Visualização dos dataframes", expanded=False):

        tab_df1, tab_df2, tab_df3, tab_df4, tab_df5 = st.tabs(
            [
                "Visão do dataframe original",
                "Visão das informações duplicadas",
                "Tratamento de NaN's e Outliers",
                "Feature Engineering",
                "Feature Selection",
            ]
        )

        with tab_df1:
            st.markdown("Conjunto de dados na forma original")
            st.dataframe(st.session_state.df_completo)
            st.write(st.session_state.df_completo.shape)

        with tab_df2:
            st.markdown("Demonstração de duplicatas existentes no conjunto de dados")
            st.dataframe(df_duplicatas.sort_values(by="id_cliente"))
            st.write(df_duplicatas.shape)

            if st.button("Drop duplicates", key="btn_drop_duplicates"):
                st.session_state.df_completo = (
                    st.session_state.df_completo.drop_duplicates(
                        subset="id_cliente", keep="first"
                    )
                )
                st.session_state.dados_duplicados = True
                st.rerun()  # Força atualização

        with tab_df3:
            # ... conteúdo tratamento NaN e outliers ...
            # Checkbox para exibir linhas com valores ausentes
            if st.checkbox("Exibir dataframe de valores ausentes"):
                st.dataframe(st.session_state.df_completo[mask])
            # Checkbox para exibir tabela com contagem de valores ausentes
            if st.checkbox("Exibir tabela com quantidades de valores ausentes"):
                st.table(st.session_state.tabela_de_ausentes)
            # Cria duas colunas para botões de tratamento
            col1, col2 = st.columns([1, 1])
            # Botão para preencher valores ausentes com -1
            if col1.button("Preencher valores ausentes com -1"):
                st.session_state.df_completo.loc[mask, "tempo_emprego"] = -1
                st.session_state.dados_tratados = True

            # Botão para apagar linhas com valores ausentes
            if col2.button("Apagar linhas com valores ausentes"):
                st.session_state.df_completo = st.session_state.df_completo[~mask]
                st.session_state.dados_tratados = True
            # Checkbox para exibir dataframe com outliers detectados
            if st.checkbox(
                """Exibir dataframe de outliers, detectados pelo método "Isolation Forest" """
            ):
                st.dataframe(st.session_state.df_completo_com_outliers)

        with tab_df4:
            st.markdown(
                "Construção de três novas variáveis para complementar os dados existentes no dataframe"
            )
            # Exibe as novas variáveis criadas no dataframe
            st.dataframe(
                st.session_state.df_completo[
                    ["possui_dependentes", "idade_jovem", "idade_x_tempo_emprego"]
                ]
            )

        with tab_df5:
            st.markdown(
                "Seleção de variáveis numéricas fundamentais para nosso modelo através da função `stepwise_selection`",
                unsafe_allow_html=True,
            )

            if st.button("Realizar Feature Selection"):
                st.session_state.feature_selection_button = True

            st.write("""As variáveis consideradas relevantes o suficiente foram:""")

            if st.session_state.feature_selection_button == True:
                st.markdown(
                    [f"`{variavel}`" for variavel in result], unsafe_allow_html=True
                )

    # SEGUNDO expander - Visualizações gráficas (SEPARADO)
    with st.expander("Visualizações gráficas das variáveis disponíveis"):
        # MUDAR NOMES das tabs internas:
        tab_viz1, tab_viz2 = st.tabs(
            [
                "Visualização da distribuição de cada variável",
                "Visualização Bivariada entre variáveis categóricas e numéricas",
            ]
        )

        with tab_viz1:
            # Remove colunas não relevantes para visualização
            st.session_state.df_visualizacao.drop(
                columns=["id_cliente", "data_ref"], inplace=True, errors="ignore"
            )
            # Cria um dataframe com informações sobre cada coluna
            st.session_state.var_info = pd.DataFrame(
                [
                    {
                        "Nome da coluna": nome,
                        "Quantidade de valores únicos": col.nunique(),
                        "Faixa": "Contínua" if col.nunique() > 9 else "Categórica",
                    }
                    for nome, col in st.session_state.df_visualizacao.items()
                ]
            )

            # Cria três colunas para controles de visualização
            col1, col2, col3 = st.columns([2, 1, 1])
            # Checkbox para visualizar tabela de variáveis categóricas e contínuas
            if col1.checkbox("Visualizar variáveis categóricas e contínuas"):
                col1.markdown(
                    "Variáveis categóricas definidas abaixo de 9 valores únicos"
                )
                col1.table(st.session_state.var_info)

            # Checkbox para mostrar distribuição de valores únicos de uma variável categórica
            if st.checkbox("Demonstrar distribuição de valores únicos"):
                distribuicao_var = st.selectbox(
                    "Selecione uma variável",
                    st.session_state.var_info.loc[
                        st.session_state.var_info["Faixa"] == "Categórica",
                        "Nome da coluna",
                    ],
                )
                # Chama função para mostrar proporção de categorias
                visualizacao_freq = proporcao_de_categorias(
                    dataframe=st.session_state.df_visualizacao, var=distribuicao_var
                )
            # Checkbox para análise bivariada entre variável categórica e contínua
            if st.checkbox("Análise Bivariada"):
                variavel_numerica_graph = st.selectbox(
                    "Selecione uma variável",
                    st.session_state.var_info.loc[
                        st.session_state.var_info["Faixa"] == "Contínua",
                        "Nome da coluna",
                    ],
                )

                categorica_var_graph = st.selectbox(
                    "Selecione uma variável",
                    st.session_state.var_info.loc[
                        st.session_state.var_info["Faixa"] == "Categórica",
                        "Nome da coluna",
                    ],
                    key="categorica_var_graph",
                )

                graph_choice = st.selectbox(
                    "Selecione uma plotagem para o gráfico",
                    ["boxplot", "violinplot", "barplot"],
                    key="graph_choice",
                )
                # Chama função para gerar gráfico bivariado
                grafico_bivariado = descritiva_bivariada(
                    setup=graph_choice,
                    categorica=categorica_var_graph,
                    numerica=variavel_numerica_graph,
                    df=st.session_state.df_visualizacao,
                )

        with tab_viz2:
            # ... conteúdo visualizações bivariadas ...
            pass

with main_tab_two:
    # Expander para regressão linear
    with st.expander("Modelos Linha de Regressão", expanded=False):
        st.session_state.selecao_var_explicativa = st.multiselect(
            "Selecione as variáveis explicativas", X_treino.columns.to_list()
        )

        if st.button("Gerar modelo de regressão linear", key="btn_linear_reg"):
            if st.session_state.selecao_var_explicativa:
                formula = "+".join(st.session_state.selecao_var_explicativa)
                st.session_state.modelo_reg = smf.ols(
                    f"renda ~ {formula}", data=df_treino
                ).fit()
                st.session_state.generate_model_button = True
            else:
                st.warning("Selecione pelo menos uma variável explicativa!")

        if (
            st.session_state.generate_model_button
            and st.session_state.modelo_reg is not None
        ):
            st.markdown("**Sumário do modelo:**")
            st.text(st.session_state.modelo_reg.summary())

    # Expander para árvore de regressão (SÓ EXECUTA quando solicitado)
    if (st.session_state.dados_duplicados) & (st.session_state.dados_tratados) == True:

        with st.expander("Modelo de Árvore de Regressão", expanded=False):

            if st.button("Treinar Árvore de Regressão", key="btn_train_tree"):
                if not (
                    st.session_state.dados_duplicados
                    and st.session_state.dados_tratados
                ):
                    st.warning(
                        "⚠️ Primeiro trate os dados duplicados e missing values na aba de Análise!"
                    )
                    st.info(
                        "Use os botões 'Drop duplicates' e 'Preencher valores ausentes com -1'"
                    )

                if (st.session_state.dados_duplicados) & (
                    st.session_state.dados_tratados
                ):

                    # 1. SEPARAR colunas por tipo
                    colunas_sem_renda = X_treino.columns.copy()
                    colunas_categoricas = [
                        "tipo_renda",
                        "educacao",
                        "estado_civil",
                        "tipo_residencia",
                    ]
                    colunas_categoricas = [
                        col for col in colunas_categoricas if col in colunas_sem_renda
                    ]
                    colunas_numericas_booleanas = [
                        col
                        for col in colunas_sem_renda
                        if col not in colunas_categoricas
                    ]

                    # 2. PROCESSAR dados
                    X_treino_numerico = X_treino[colunas_numericas_booleanas].copy()

                    if colunas_categoricas:
                        X_treino_categorico_dummies = pd.get_dummies(
                            X_treino[colunas_categoricas], drop_first=True
                        ).astype(int)
                    else:
                        X_treino_categorico_dummies = pd.DataFrame()

                    X_treino_processed = pd.concat(
                        [X_treino_numerico, X_treino_categorico_dummies], axis=1
                    ).fillna(0)

                    # 3. TREINAR modelo
                    arvore_reg = modelando_arvore_regressao(
                        X_treino_processed, y_treino
                    )

                    # 4. SALVAR no session_state para exibir resultados
                    st.session_state.arvore_reg = arvore_reg
                    st.session_state.X_treino_processed = X_treino_processed
                    st.session_state.y_treino = y_treino

                    # 5. EXIBIR RESULTADOS
                    st.success("✅ Modelo treinado com sucesso!")

                    # EXIBIR RESULTADOS se o modelo foi treinado
                    if "arvore_reg" in st.session_state:
                        st.subheader("Melhores Hiperparâmetros Encontrados")
                        st.json(st.session_state.arvore_reg.best_params_)

                        st.subheader("Desempenho do Modelo")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(
                                "Melhor R² (Validação Cruzada)",
                                f"{st.session_state.arvore_reg.best_score_:.4f}",
                            )

                        with col2:
                            r2_treino = st.session_state.arvore_reg.score(
                                st.session_state.X_treino_processed,
                                st.session_state.y_treino,
                            )
                            st.metric("R² no Treino", f"{r2_treino:.4f}")

                # ✅ NOVA SEÇÃO PARA PREVISÕES
        with st.expander("🔮 Fazer Previsões com o Modelo", expanded=True):
            st.subheader("Simulador de Previsão de Renda")

            # Criar formulário para entrada de dados
            with st.form("form_previsao"):
                col1, col2 = st.columns(2)

                with col1:
                    idade = st.slider("Idade", 18, 100, 35)
                    tempo_emprego = st.slider("Tempo no Emprego (anos)", 0, 50, 5)
                    qtd_filhos = st.number_input("Quantidade de Filhos", 0, 10, 0)
                    posse_imovel = st.selectbox("Possui Imóvel?", ["Não", "Sim"])

                with col2:
                    tipo_renda = st.selectbox(
                        "Tipo de Renda",
                        [
                            "Assalariado",
                            "Empresário",
                            "Pensionista",
                            "Servidor público",
                        ],
                    )
                    educacao = st.selectbox(
                        "Educação",
                        [
                            "Secundário",
                            "Superior incompleto",
                            "Superior completo",
                            "Pós graduação",
                        ],
                    )
                    estado_civil = st.selectbox(
                        "Estado Civil",
                        ["Solteiro", "Casado", "União", "Divorciado", "Viúvo"],
                    )
                    tipo_residencia = st.selectbox(
                        "Tipo de Residência",
                        ["Casa", "Apartamento", "Com os pais", "Outros"],
                    )

                # Botão para fazer previsão
                submitted = st.form_submit_button("Prever Renda")

            # Processar previsão quando o formulário for submetido
            if submitted:
                # Criar dataframe com os dados de entrada
                input_data = pd.DataFrame(
                    {
                        "idade": [idade],
                        "tempo_emprego": [tempo_emprego],
                        "qtd_filhos": [qtd_filhos],
                        "posse_de_imovel": [1 if posse_imovel == "Sim" else 0],
                        "posse_de_veiculo": [
                            0
                        ],  # Valor padrão ou adicione campo no formulário
                        "tipo_renda": [tipo_renda],
                        "educacao": [educacao],
                        "estado_civil": [estado_civil],
                        "tipo_residencia": [tipo_residencia],
                        "possui_dependentes": [1 if qtd_filhos > 0 else 0],
                        "idade_jovem": [1 if idade <= 25 else 0],
                        "idade_x_tempo_emprego": [idade / (tempo_emprego + 1)],
                    }
                )

                # Pré-processar os dados igual foi feito no treino
                input_processed = preprocess_input(
                    input_data, st.session_state.X_treino_processed.columns
                )

                # Fazer previsão
                try:
                    previsao = st.session_state.arvore_reg.predict(input_processed)[0]
                    st.success(f"### Previsão de Renda: R$ {previsao:,.2f}")

                    # Adicionar algumas métricas contextuais
                    st.metric("Previsão Mensal", f"R$ {previsao:,.2f}")
                    st.metric("Previsão Anual", f"R$ {previsao * 12:,.2f}")

                except Exception as e:
                    st.error(f"Erro na previsão: {e}")
