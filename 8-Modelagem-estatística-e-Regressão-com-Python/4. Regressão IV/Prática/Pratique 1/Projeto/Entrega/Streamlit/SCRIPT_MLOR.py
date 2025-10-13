# Importa a biblioteca pandas para manipulação de dados em DataFrames
import pandas as pd

# Importa a biblioteca numpy para operações numéricas
import numpy as np

# Importa a função para modelos lineares do statsmodels usando fórmulas
import statsmodels.formula.api as smf

# Importa seaborn para visualização de dados estatísticos
import seaborn as sns

# Importa o módulo pyplot do matplotlib para criação de gráficos
import matplotlib.pyplot as plt

# Importa a API principal do statsmodels para análise estatística
import statsmodels.api as sm

# Importa função para dividir os dados em treino e teste
from sklearn.model_selection import train_test_split

# Importa a função para cálculo do VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant

# Importa unicodedata para normalização de strings (remoção de acentos)
import unicodedata

# Importa função para plotar gráficos de influência em regressão
from statsmodels.graphics.regressionplots import influence_plot

# Importa função para cálculo do erro quadrático médio
from sklearn.metrics import mean_squared_error

# Importa streamlit para criação de aplicações web interativas
import streamlit as st

# Importa io para manipulação de arquivos em memória
import io


def main():
    # Configura o título da aplicação
    st.set_page_config(
        page_title="Projeto para criação de clusters",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="varig_icon.png",
    )
    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">

    <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 3em;'>
        <strong>Regularização do conjunto de dados</strong>
    </h1>
    """,
        unsafe_allow_html=True,
    )


def performance_de_teste(_y_test, _y_pred, _modelo):
    # Cálculo de R-Quadrado (coeficiente de determinação) na base de teste
    _tss_test = ((_y_test - _y_test.mean()) ** 2).sum()  # Soma dos quadrados totais
    _rss_test = ((_y_test - _y_pred) ** 2).sum()  # Soma dos quadrados dos resíduos
    _r_quadrado_test = (
        1 - _rss_test / _tss_test
    )  # R²: proporção da variância explicada pelo modelo

    # Cálculo de R-Quadrado Ajustado na base de teste
    _qtd_variaveis_explicativas = (
        len(_modelo.params) - 1
    )  # Número de variáveis explicativas (exclui o intercepto)
    _r_quadrado_ajustado_teste = 1 - (1 - _r_quadrado_test) * (len(_y_test) - 1) / (
        len(_y_test) - _qtd_variaveis_explicativas - 1
    )

    # Exibe os resultados formatados
    print(f"R-quadrado: {_r_quadrado_test:.2%}")
    print(f"R-quadrado Ajustado: {_r_quadrado_ajustado_teste:.2%}")
    return (
        _r_quadrado_test,
        _r_quadrado_ajustado_teste,
    )  # Retorna os valores para uso posterior


# Função para criar dataframe VIF e realizar a filtragem de variáveis que possuírem um valor acima do estipulado nos parâmetros
def vif_filter(X, limite=10):
    """
    Filtra variáveis com alto fator de inflação de variância (VIF) de um DataFrame.

    Parâmetros:
    X (pd.DataFrame): DataFrame contendo apenas variáveis numéricas independentes.
    limite (float): Valor limite de VIF para remoção de variáveis (padrão=10).

    Retorna:
    X_filtrado (pd.DataFrame): DataFrame com variáveis remanescentes após filtragem.
    removed_features (list): Lista de tuplas (variável, VIF) removidas.
    remaining_vif (pd.DataFrame): DataFrame com VIF das variáveis remanescentes.
    """
    X_filtrado = X.copy()  # Cria uma cópia do DataFrame original
    removed_features = []  # Lista para armazenar variáveis removidas
    vif_scores = {}  # Dicionário para armazenar os VIFs

    while True:
        # Adiciona constante para cálculo do VIF
        X_with_const = add_constant(X_filtrado)

        # Calcula o VIF para cada variável (incluindo a constante)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_with_const.columns
        vif_data["VIF"] = [
            vif(X_with_const.values, i) for i in range(X_with_const.shape[1])
        ]

        # Remove a constante da análise
        vif_data = vif_data[vif_data["feature"] != "const"]

        # Salva os VIFs calculados
        for _, row in vif_data.iterrows():
            vif_scores[row["feature"]] = row["VIF"]

        # Verifica o maior VIF
        max_vif = vif_data["VIF"].max()
        if max_vif <= limite:
            break  # Sai do loop se todos os VIFs estiverem abaixo do limite

        # Remove a variável com maior VIF
        feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
        removed_features.append((feature_to_remove, max_vif))
        X_filtrado = X_filtrado.drop(columns=[feature_to_remove])

        # Se todas as variáveis forem removidas, lança erro
        if X_filtrado.shape[1] == 0:
            raise ValueError(
                "Todas as variáveis foram removidas - limite pode estar muito baixo"
            )

    # Monta DataFrame final com VIF das variáveis remanescentes
    remaining_vif = vif_data[vif_data["feature"].isin(X_filtrado.columns)]
    remaining_vif = remaining_vif.sort_values("VIF", ascending=False)

    return X_filtrado, removed_features, remaining_vif


# Função para remover caracteres especiais e adicionar "_" nos espaços
def padronizar_nome(var):
    """
    Remove caracteres especiais de uma string e substitui espaços por underline.

    Parâmetros:
    var (str): Nome da variável a ser padronizado.

    Retorna:
    str: Nome padronizado, sem acentos e com espaços substituídos por "_".
    """
    # Normaliza a string para remover acentos e caracteres especiais
    var = unicodedata.normalize("NFKD", var).encode("ASCII", "ignore").decode("ASCII")
    return var.replace(" ", "_")  # Substitui espaços por underline


def esp_vs_obs(
    data, res, X, y_pred, y_true, transformar=False, amostragem=False, tamanho=1000
):
    # Obtém as previsões do modelo OLS
    pred_ols = res.get_prediction()
    pred_summary = pred_ols.summary_frame()

    # Extrai intervalos de confiança para observações e médias
    iv_l = pred_summary["obs_ci_lower"]
    iv_u = pred_summary["obs_ci_upper"]
    m_l = pred_summary["mean_ci_lower"]
    m_u = pred_summary["mean_ci_upper"]

    # Se amostragem for True, utiliza apenas as primeiras 'tamanho' linhas
    if amostragem:
        data = data.head(tamanho)
        y_pred = y_pred.head(tamanho)
        iv_l = iv_l.head(tamanho)
        iv_u = iv_u.head(tamanho)
        m_l = m_l.head(tamanho)
        m_u = m_u.head(tamanho)

    # Cria uma cópia do dataframe ordenado pelo eixo X
    df_plot = data.copy()
    df_plot["y_pred"] = y_pred
    df_plot["iv_l"] = iv_l
    df_plot["iv_u"] = iv_u
    df_plot["m_l"] = m_l
    df_plot["m_u"] = m_u
    # Se transformar=True, aplica log na variável resposta real
    df_plot["y_real"] = np.log(data[y_true]) if transformar else data[y_true]
    df_plot = df_plot.sort_values(by=X)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter dos pontos observados
    ax.plot(df_plot[X], df_plot["y_pred"], "o", label="Dados observados", alpha=0.5)

    # Linhas conectando em ordem
    ax.plot(
        df_plot[X],
        df_plot["y_real"],
        "b-",
        label="Média da simulação (log)" if transformar else "Média da simulação",
    )
    ax.plot(df_plot[X], df_plot["y_pred"], "r--", label="Estimativa OLS")
    ax.plot(df_plot[X], df_plot["iv_u"], "r--", label="Banda de confiança para y")
    ax.plot(df_plot[X], df_plot["iv_l"], "r--")
    ax.plot(df_plot[X], df_plot["m_u"], "g:", label="Banda de confiança para média")
    ax.plot(df_plot[X], df_plot["m_l"], "g:")

    ax.set_xlabel(X)
    ax.set_ylabel(y_true)
    ax.set_title("Expectativa vs Observado")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


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
            model = sm.OLS(
                y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))
            ).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = (
                new_pval.idxmin()
            )  # Use idxmin() em vez de argmin() para garantir o nome da coluna
            included.append(best_feature)
            changed = True
            if verbose:
                print("Add  {:30} with p-value {:.6}".format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = (
                pvalues.idxmax()
            )  # Use idxmax() em vez de argmax() para garantir o nome da coluna
            included.remove(worst_feature)
            if verbose:
                print("Drop {:30} with p-value {:.6}".format(worst_feature, worst_pval))
        if not changed:
            break
    return included


main()  # Executa a função principal, caso exista

st.markdown("---")


# === CARREGAMENTO DO CSV PELO USUÁRIO ===
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")


if uploaded_file is not None and "df" not in st.session_state:
    # Leitura do CSV
    df_original = pd.read_csv(uploaded_file)
    st.session_state.df = df_original.copy()

df = st.session_state.get("df")
# === SEÇÃO 1: VALORES AUSENTES ===

if uploaded_file is not None:
    with st.expander(
        "🧹 1. Tratamento de Valores Ausentes e Definição de tamanho da amostragem",
        expanded=False,
    ):
        # Divide a tela em 3 colunas, com a central (col2) maior
        col1, col2, col3 = st.columns(([1, 2, 1]))
        tabela_de_missings = col2.empty()
        tabela_de_missings.table(df.isna().sum())

        #  Lista de colunas com valores ausentes
        variaveis_missing = [col for col, val in df.isna().sum().items() if val > 0]

        variável_para_tratamento = st.multiselect(
            "Dados faltantes a serem tratadas", variaveis_missing
        )
        # Botão para preencher valores ausentes com a média
        if st.button("Preencher dados faltantes com a média dos valores"):
            for var in variável_para_tratamento:
                df[var].fillna(df[var].mean(), inplace=True)  # forma mais elegante

            st.success(
                "Todos os dados ausentes foram substítuidos pela média da coluna"
            )

        # Atualiza a exibição da tabela de missings
        tabela_de_missings.table(df.isna().sum())  # reexibe com os valores atualizados

        # === SEÇÃO 3: SEPARAÇÃO DO CONJUNTO DE TREINO ===

        # Obtém a lista de todas as colunas do DataFrame df
        variaveis = df.columns.to_list()

        # Remove as colunas 'data_ref' e 'index' da lista de variáveis, pois não serão usadas nas análises
        variaveis = [x for x in variaveis if x not in ["data_ref", "index"]]

        # Define a variável dependente (target) como a nova coluna 'log_renda'
        y = df["renda"]

        # Define as variáveis independentes removendo as colunas 'renda' e 'log_renda'
        X = df[variaveis].drop(columns=["renda"])

        # Separa os dados em treino e teste (60% treino, 40% teste, com semente fixa para reprodutibilidade)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=100
        )

        # Junta as variáveis independentes e dependente para formar o dataframe de treino
        df_treino = X_train.join(y_train)

        if "df_resid_check" not in st.session_state:
            st.session_state.df_resid_check = None

        # === SEÇÃO 4: AMOSTRAGEM PARA RESÍDUOS ===

        numero_de_linhas = st.number_input(
            "Descreva o número máximo de linhas para o conjunto de treino",
            help="Limita o número de linhas do conjunto de treino para análise",
            step=1000,
            min_value=1000,
            max_value=len(df_treino),
            value=min(30000, len(df_treino)),  # valor padrão
        )

        if st.button("Aplique o limite de linhas"):
            st.session_state.df_resid_check = df_treino.sample(
                numero_de_linhas, random_state=100
            ).reset_index(drop=True)

            st.success(f"Amostra de {numero_de_linhas} linhas criada com sucesso!")

    st.markdown("---")
    df_resid_check = st.session_state.get("df_resid_check")

    if st.session_state.df_resid_check is not None:
        # Seleciona as variáveis independentes, excluindo 'renda', 'index' e 'data_ref'
        variaveis_independentes = [
            var
            for var in st.session_state.df_resid_check.columns.to_list()
            if var not in ["renda", "index", "data_ref"]
        ]

        # Monta a fórmula para o modelo de regressão linear
        formula = "renda ~" + "+".join(variaveis_independentes)

        # Ajusta o modelo de regressão linear usando o DataFrame de verificação de resíduos
        modelo = smf.ols(formula, data=st.session_state.df_resid_check).fit()

        fig_original, ax_original = plt.subplots(
            figsize=(8, 5)
        )  # Cria uma figura e eixo

        y_pred_regressao_1 = modelo.predict(
            st.session_state.df_resid_check[variaveis_independentes]
        )
        # Plota o gráfico dos resíduos (diferença entre valores observados e previstos) em relação aos valores previstos
        distribuicao_de_residuos = sns.residplot(
            x=y_pred_regressao_1,  # valores previstos pelo modelo
            y=np.log(
                st.session_state.df_resid_check["renda"] + 1
            ),  # log da renda observada
            data=st.session_state.df_resid_check,
            lowess=True,  # suavização local
            scatter_kws={"alpha": 0.5},  # transparência dos pontos
            line_kws={
                "color": "red",
                "lw": 1,
                "alpha": 0.8,
            },  # estilo da linha de tendência
            ax=ax_original,
        )

        ax_original.set_title("Resíduos sem tratamento")

        # Salva a figura como imagem (buffer)
        buf_original = io.BytesIO()
        fig_original.savefig(buf_original, format="png", bbox_inches="tight", dpi=100)
        buf_original.seek(0)
        st.markdown("---")

        # TRABALHANDO COM A CRIAÇÃO COM UM MODELO DE REGRESSÃO SEGMENTADA
        # Definir pontos de corte para testar
        C1_values = np.linspace(
            start=df["tempo_emprego"].min(), stop=df["tempo_emprego"].max(), num=5
        )  # Reduzindo para 5 pontos para cada variável

        C2_values = np.linspace(
            start=df["qtd_filhos"].min(), stop=df["qtd_filhos"].max(), num=5
        )

        C3_values = np.linspace(start=df["idade"].min(), stop=df["idade"].max(), num=5)

        # Lista para armazenar resultados
        resultados = []

        # Testar combinações de pontos de corte
        for c1 in C1_values:
            # Criar segmentos para tempo_emprego
            st.session_state.df_resid_check["X1_1"] = np.where(
                st.session_state.df_resid_check["tempo_emprego"] <= c1,
                st.session_state.df_resid_check["tempo_emprego"],
                c1,
            )
            st.session_state.df_resid_check["X1_2"] = np.where(
                st.session_state.df_resid_check["tempo_emprego"] > c1,
                st.session_state.df_resid_check["tempo_emprego"] - c1,
                0,
            )

            for c2 in C2_values:
                # Criar segmentos para qtd_filhos
                st.session_state.df_resid_check["X2_1"] = np.where(
                    st.session_state.df_resid_check["qtd_filhos"] <= c2,
                    st.session_state.df_resid_check["qtd_filhos"],
                    c2,
                )
                st.session_state.df_resid_check["X2_2"] = np.where(
                    st.session_state.df_resid_check["qtd_filhos"] > c2,
                    st.session_state.df_resid_check["qtd_filhos"] - c2,
                    0,
                )

                for c3 in C3_values:
                    # Criar segmentos para idade
                    st.session_state.df_resid_check["X3_1"] = np.where(
                        st.session_state.df_resid_check["idade"] <= c3,
                        st.session_state.df_resid_check["idade"],
                        c3,
                    )
                    st.session_state.df_resid_check["X3_2"] = np.where(
                        st.session_state.df_resid_check["idade"] > c3,
                        st.session_state.df_resid_check["idade"] - c3,
                        0,
                    )

                    # Ajustar modelo
                    formula = "I(np.log(renda+1)) ~ I(np.log(X1_1 + 2)) + I(np.log(X1_2 + 2)) + X2_1 + X2_2 + I(np.log(X3_1 + 2)) + I(np.log(X3_2 + 2))"
                    try:
                        modelo = smf.ols(
                            formula, data=st.session_state.df_resid_check
                        ).fit()

                        # Armazenar resultados
                        resultados.append(
                            {
                                "corte_tempo_emprego": c1,
                                "corte_qtd_filhos": c2,
                                "corte_idade": c3,
                                "R2": modelo.rsquared,
                                "AIC": modelo.aic,
                                "BIC": modelo.bic,
                            }
                        )
                    except:
                        continue

        # === SEÇÃO 5: DEMONSTRAÇÃO DAS POSIÇÕES DE CORTE PARA REGRESSÃO SEGMENTADA ===

        df_resultados = pd.DataFrame(resultados)

        st.write(
            "Para linearizar os resíduos, as variáveis numéricas passam pela transformação logarítimica, se somando com 2, o mesmo para variável de interesse. Também é implementada a técnica de regressão segmentada para o modelo a ser construído"
        )

        st.write(
            "Elaboração de um dataframe com os valores de corte para a regressão segmentada"
        )
        # Mostrar melhores combinações
        with st.expander("Visualize os parâmetros de corte", expanded=False):
            # Converter para DataFrame
            st.dataframe(df_resultados.sort_values("R2", ascending=False).head(10))

        C1 = 10.815068
        C2 = 3.5
        C3 = 45

        ### CRIANDO DOIS SEGMENTOS PARA A VARIÁVEL TEMPO_EMPREGO
        # Cria a variável segmentada X1_1: igual a X1 se X1 <= C1, senão igual a C1
        st.session_state.df_resid_check["X1_1"] = (
            st.session_state.df_resid_check["tempo_emprego"] <= C1
        ) * st.session_state.df_resid_check["tempo_emprego"] + (
            st.session_state.df_resid_check["tempo_emprego"] > C1
        ) * C1

        # Cria a variável segmentada X1_2: igual a X1 - C1 se X1 > C1, senão zero
        st.session_state.df_resid_check["X1_2"] = (
            st.session_state.df_resid_check["tempo_emprego"] > C1
        ) * (st.session_state.df_resid_check["tempo_emprego"] - C1)

        ### CRIANDO DOIS SEGMENTOS PARA QTD_FILHOS
        # Cria a variável segmentada X2_1: igual a qtd_filhos se qtd_filhos <= C2, senão igual a C2
        st.session_state.df_resid_check["X2_1"] = (
            st.session_state.df_resid_check["qtd_filhos"] <= C2
        ) * st.session_state.df_resid_check["qtd_filhos"] + (
            st.session_state.df_resid_check["qtd_filhos"] > C2
        ) * C2

        # Cria a variável segmentada X1_2: igual a qtd_filhos - C2 se X1 > C2, senão zero
        st.session_state.df_resid_check["X2_2"] = (
            st.session_state.df_resid_check["qtd_filhos"] > C2
        ) * (st.session_state.df_resid_check["qtd_filhos"] - C2)

        ### CRIANDO DOIS SEGMENTOS PARA A VARIÁVEL IDADE
        # Cria a variável segmentada X3_1: igual a idade se idade <= C3, senão igual a C3
        st.session_state.df_resid_check["X3_1"] = (
            st.session_state.df_resid_check["idade"] <= C3
        ) * st.session_state.df_resid_check["idade"] + (
            st.session_state.df_resid_check["idade"] > C3
        ) * C3

        # Cria a variável segmentada X1_2: igual a X1 - C3 se X1 > C3, senão zero
        st.session_state.df_resid_check["X3_2"] = (
            st.session_state.df_resid_check["idade"] > C3
        ) * (st.session_state.df_resid_check["idade"] - C3)

        # Separa os dados em treino e teste (60% treino, 40% teste, com semente fixa para reprodutibilidade)
        X_train_, X_test_, y_train_, y_test_ = train_test_split(
            st.session_state.df_resid_check.drop(columns=("renda")),
            st.session_state.df_resid_check.renda,
            test_size=0.4,
            random_state=100,
        )

        # Ajusta o modelo de regressão linear segmentado usando as transformações logarítmicas das variáveis segmentadas
        res_segmentada = smf.ols(
            "I(np.log(renda+1)) ~ I(np.log(X1_1 + 2)) + I(np.log(X1_2 + 2)) + I(np.log(X2_1 + 2)) + I(np.log(X2_2 + 2))+ I(np.log(X3_1 + 2))  + I(np.log(X3_2 + 2))",
            data=X_train_.join(y_train_),
        ).fit()

        y_pred_regressao_2 = res_segmentada.predict(
            st.session_state.df_resid_check[
                ["X1_1", "X1_2", "X2_1", "X2_2", "X3_1", "X3_2"]
            ]
        )

        ### ATUALIZANDO A DISTRIBUIÇÃO DOS RESÍDUOS

        fig_tratado, ax_tratado = plt.subplots(figsize=(8, 5))  # Cria uma figura e eixo

        # Plota o gráfico dos resíduos (diferença entre valores observados e previstos) em relação aos valores previstos
        distribuicao_de_residuos = sns.residplot(
            x=y_pred_regressao_2,
            y=np.log(
                st.session_state.df_resid_check["renda"] + 1
            ),  # log da renda observada
            data=st.session_state.df_resid_check,
            lowess=True,  # suavização local
            scatter_kws={"alpha": 0.5},  # transparência dos pontos
            line_kws={
                "color": "red",
                "lw": 1,
                "alpha": 0.8,
            },  # estilo da linha de tendência
            ax=ax_tratado,
        )
        ax_tratado.set_title("Resíduos após tratamento")

        # Salva a figura como imagem (buffer)
        buf_tratado = io.BytesIO()
        fig_tratado.savefig(buf_tratado, format="png", bbox_inches="tight", dpi=100)
        buf_tratado.seek(0)

        # Mostrar seção de resíduos

        with st.expander("Resíduos do conjunto de dados", expanded=False):
            tipo_grafico = st.radio(
                "Escolha o gráfico de resíduos:",
                ["Original", "Transformado"],
                index=0,  # valor padrão
                horizontal=True,
            )
            fig_histograma, ax_histograma = plt.subplots(figsize=(8, 5))
            sns.histplot(res_segmentada.resid, bins=30, ax=ax_histograma)
            buf_histograma = io.BytesIO()
            fig_histograma.savefig(
                buf_histograma, format="png", bbox_inches="tight", dpi=100
            )
            buf_histograma.seek(0)
            # Exibe o gráfico correspondente
            if tipo_grafico == "Transformado":
                st.image(buf_tratado, width=800, caption="Resíduos após tratamento")
            else:
                st.image(buf_original, width=800, caption="Resíduos do modelo original")

            if st.checkbox("Exibir histograma da distribuição de resíduos"):

                st.image(
                    buf_histograma,
                    width=800,
                    caption="Distribuição dos resíduos pós tratamento",
                )

        st.markdown("---")
        # === SEÇÃO 6: VISUALIZANDO OS OUTILIERS ===
        st.markdown("## Buscando os outliers")
        st.markdown(
            "Foi constatado que não havia necessidade para o tratamento de outliers, se lastreando principalmente no gráfico de influência"
        )
        fig_scatter_outlier, ax_scatter_outlier = plt.subplots(figsize=(8, 5))
        st_res = res_segmentada.outlier_test()
        sns.scatterplot(x=y_pred_regressao_2, y=st_res.student_resid)
        buf_outliers_scatter = io.BytesIO()
        fig_scatter_outlier.savefig(
            buf_outliers_scatter, format="png", bbox_inches="tight", dpi=100
        )
        buf_outliers_scatter.seek(0)

        fig_boxplot, ax_boxplot = plt.subplots(1, 3, figsize=(8, 4))

        sns.boxplot(y="renda", data=df_resid_check, ax=ax_boxplot[0])
        sns.boxplot(y="tempo_emprego", data=df_resid_check, ax=ax_boxplot[1])
        sns.boxplot(y=res_segmentada.resid, data=df_resid_check, ax=ax_boxplot[2])

        ax_boxplot[0].set_ylabel("renda")
        ax_boxplot[1].set_ylabel("Tempo Emprego")
        ax_boxplot[2].set_ylabel("Resíduo")

        buf_boxplot = io.BytesIO()
        fig_boxplot.savefig(buf_boxplot, format="png", bbox_inches="tight", dpi=100)
        buf_boxplot.seek(0)

        # influence = res_segmentada.get_influence()
        # summary_inf = influence.summary_frame()

        # # Define critérios
        # outlier_condition = (np.abs(summary_inf["student_resid"]) > 3) | (
        #     summary_inf["cooks_d"] > 4 / len(df_treino)
        # )

        # # Descobre os valores "extremos"
        # df_outliers = X_train_.join(y_test_)[outlier_condition]

        # # Calcula influência
        # influence = res_segmentada.get_influence()
        # summary_frame = influence.summary_frame()
        # top_points = summary_frame.sort_values("cooks_d", ascending=False).head(5)

        # # Plot básico
        # fig_influence, ax_influence = plt.subplots(figsize=(8, 6))
        # influence_plot(
        #     res_segmentada, ax=ax_influence, criterion="cooks", alpha=0.5, labels=None
        # )

        # for text in ax_influence.texts:
        #     text.set_visible(False)

        # # Adiciona apenas os 5 pontos mais influentes
        # for i in top_points.index:
        #     x = influence.hat_matrix_diag[i]
        #     y = influence.resid_studentized_external[i]
        #     ax_influence.annotate(i, (x, y), fontsize=8, color="red")

        # buf_influencia = io.BytesIO()
        # fig_influence.savefig(buf_influencia, format="png", bbox_inches="tight", dpi=100)
        # buf_influencia.seek(0)
        with st.expander(
            "Gráficos para outliers",
            expanded=False,
        ):
            col1, col2 = st.columns([1, 1])
            col1.image(
                buf_outliers_scatter, width=800, caption="Distribuição dos pontos"
            )
            col2.image(buf_boxplot, width=800, caption="Quantificação dos Outliers")
            # col1.image(
            #     buf_influencia,
            #     width=800,
            #     caption="Gráfico para captar a Influência dos outliers no modelo",
            # )

        st.markdown("---")
        st.markdown("## Diagnosticando e tratando Multicolinearidade")
    # Calcula a matriz de correlação de Spearman apenas para as variáveis numéricas do dataframe 'df'
    correlacao_spearmen = (
        df[variaveis]
        .select_dtypes(include=["float64", "int64"])
        .corr(method="spearman")
    )

    # Substitui os valores da diagonal principal por NaN para facilitar a visualização (evita destacar a correlação perfeita de uma variável consigo mesma)
    np.fill_diagonal(correlacao_spearmen.values, np.nan)

    with st.expander(
        "Exibindo os resultados do método VIF",
        expanded=False,
    ):
        # Aplica o destaque (highlight) ao maior valor de cada linha da matriz de correlação de Spearman,
        # usando a cor preta para facilitar a visualização dos pares de variáveis com maior correlação em cada linha.
        st.markdown("**Correlação de Spearmen**")
        corr_data = st.dataframe(
            correlacao_spearmen.style.highlight_max(axis=1, color="black")
        )

        df_dummies_numericas = pd.get_dummies(
            data=df[variaveis], drop_first=True, dtype=int
        )

        # Renomeia as colunas do DataFrame aplicando a função de padronização de nomes
        df_dummies_numericas = df_dummies_numericas.rename(columns=padronizar_nome)

        #   Cria uma lista com todas as variáveis independentes, exceto 'renda'
        variaveis_independentes = [
            var for var in df_dummies_numericas.columns.to_list() if var != "renda"
        ]

        # Aplica a função vif_filter para remover variáveis com VIF acima de 5, retornando o dataframe ajustado, as variáveis removidas e o dataframe com os VIFs finais
        previsao_renda_vif_ajustado, variaveis_removidas, vif_frame = vif_filter(
            df_dummies_numericas.drop(columns=["renda"]), 5
        )

        vif_info = st.radio(
            "Escolher o tipo de informação obtido por VIF:",
            ["Variáveis removidas", "VIF Frame"],
            index=0,  # valor padrão
            horizontal=True,
        )

        if vif_info == "Variáveis removidas":
            st.table(variaveis_removidas)
        else:
            st.dataframe(vif_frame)

    st.markdown("---")
    st.markdown("## Ajustes finais e comparação de modelos")

    formula = "I(np.log(renda + 1)) ~ C(sexo) + C(posse_de_veiculo) + C(qtd_filhos) + C(tipo_renda) + C(estado_civil) + I(np.log( X1_1 + 2)) + I(np.log(X1_2+ 2)) + I(np.log(X3_1+ 2)) + I(np.log(X3_2+ 2))"
    # Ajusta o modelo de regressão linear segmentado usando as transformações logarítmicas das variáveis segmentadas
    reg_final = smf.ols(formula, data=X_train_.join(y_train_)).fit()
    reg_final.summary()

    with st.expander(
        "Regressão segmentada composto somente por variáveis numéricas x Regressão segmentada com variáveis numéricas e categóricas",
        expanded=False,
    ):
        comparacao_modelos = st.radio(
            "Defina qual modelo para apresentar o sumário",
            ["Variáveis numéricas", "Variáveis numéricas e categóricas"],
            index=0,
            horizontal=True,
        )
        if comparacao_modelos == "Variáveis numéricas":
            st.markdown(res_segmentada.summary().as_html(), unsafe_allow_html=True)

        else:
            st.markdown(reg_final.summary().as_html(), usafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Análise da performance dos modelos na base de teste")
    st.markdown(
        """
                Como forma de aprimorar o modelo, aplicamos a regularização por meio da técnica "stepwise selection", 
buscando otimizar a escolha das variáveis utilizadas. 
                """
    )

    X_train_clean = X_train_[["X1_1", "X1_2", "X2_1", "X2_2", "X3_1", "X3_2"]]

    results = stepwise_selection(X_train_clean, y_train_)

    st.write("\nVariáveis selecionadas:")
    st.table(results)

    r_quadrado_test, r_quadrado_test_adjusted = performance_de_teste(
        y_test_,
        res_segmentada.predict(
            X_test_[["X1_1", "X1_2", "X2_1", "X2_2", "X3_1", "X3_2"]]
        ),
        res_segmentada,
    )
    with st.expander(
        "Visualizar informações da performance dos modelos no teste", expanded=False
    ):
        st.markdown(
            f"R-Quadrado no teste para regressão segmentada{round(r_quadrado_test, 3)} "
        )
        st.markdown(
            f"R-Quadrado Ajustado no teste para regressão segmentada{round(r_quadrado_test_adjusted, 3)} "
        )
