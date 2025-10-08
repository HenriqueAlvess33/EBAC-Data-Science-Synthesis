# -----------------------------------------------------------------------
# Bibliotecas externas de manipulação, visualização e modelagem estatística
# -----------------------------------------------------------------------
import streamlit as st  # Framework para construir interfaces web interativas (widgets, layout, exibição)
import numpy as np  # Operações numéricas (arrays, funções matemáticas, log, power)
import pandas as pd  # Manipulação e análise de DataFrames
import seaborn as sns  # Visualizações estatísticas de alto nível (facilita scatter, boxplot, etc.)
import matplotlib.pyplot as plt  # Plotagem com matplotlib (base para figuras)
import plotly.express as px  # Visualizações interativas e de alto nível
import scipy.stats as stats  # Estatísticas e testes estatísticos
import plotly.graph_objects as go  # Gráficos plotly de baixo nível para controle detalhado

from io import (
    BytesIO,
)  # Buffer em memória (usado para salvar figuras e exibir no Streamlit)

# Bibliotecas para modelagem estatística baseada em fórmulas
import patsy  # Constrói matrizes de design (Y, X) a partir de fórmulas estilo R
import statsmodels.api as sm  # API geral do statsmodels (OLS, diagnósticos, etc.)
import statsmodels.formula.api as smf  # Interface de alto nível por fórmula (ex.: smf.ols)


# ------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ------------------------------------------------------
# Define título da aba, layout e ícone da aplicação Streamlit
st.set_page_config(
    page_title="Trabalhando com modelos de regressão Linear",  # Texto na aba do navegador
    layout="wide",  # Usa toda a largura disponível para o app
    initial_sidebar_state="expanded",  # Sidebar (barra lateral) inicia aberta
    page_icon="varig_icon.png",  # Ícone da aplicação (arquivo local esperado)
)

# ------------------------------------------------------
# INICIALIZAÇÃO DO ESTADO DA SESSÃO
# ------------------------------------------------------
# Inicializa chaves na sessão para armazenar escolhas/recomendações entre interações
# Isso permite guardar qual modelo foi recomendado em cada "tab" e reutilizar depois
if "modelo_recomendado_tab_1" not in st.session_state:
    st.session_state.modelo_recomendado_tab_1 = None  # Armazena melhor modelo da Tab 1

if "modelo_recomendado_tab_2" not in st.session_state:
    st.session_state.modelo_recomendado_tab_2 = None  # Armazena melhor modelo da Tab 2

if "formula" not in st.session_state:
    st.session_state.formula = None  # Armazena fórmula atual do modelo

if "previsao_renda_processed" not in st.session_state:
    st.session_state.previsao_renda_processed = None  # Armazena dataset processado


# --------------------------------------------------
# FUNÇÃO AUXILIAR: criar modelo OLS e devolver DataFrame com resíduos
# --------------------------------------------------
def criando_modelo_e_dataframe(formula, df):
    """
    Recebe:
    - formula: string no formato compatível com patsy/statsmodels (ex.: "y ~ x1 + x2")
    - df: DataFrame que contém as variáveis referidas na fórmula

    Retorna:
    - df_copy: cópia do DataFrame com uma coluna adicional 'Resíduos' (residuo do ajuste)
    - reg: objeto do modelo ajustado (resultado de sm.OLS.fit())

    Passos:
    - patsy.dmatrices constrói Y e X (matrizes) a partir da fórmula.
    - Converte Y para vetor 1D para alimentar OLS.
    - Ajusta OLS via statsmodels e guarda resíduos no DataFrame cópia.
    """
    # Constrói y (vetor resposta) e X (matriz de design) a partir da fórmula
    y, X = patsy.dmatrices(formula, df)

    # CORREÇÃO: converte X para DataFrame para preservar nomes de colunas quando desejar
    X_df = pd.DataFrame(X, columns=X.design_info.column_names)

    # Garante que y seja vetor 1D (statsmodels OLS aceita array 1D)
    y = np.asarray(y).ravel()

    # Ajusta o modelo OLS usando a matriz/DataFrame de regressores
    # Utiliza sm.OLS(y, X_df) para preservar nomes e facilitar interpretação
    reg = sm.OLS(y, X_df).fit()

    # Faz cópia do DataFrame original para não modificar fora da função
    df_copy = df.copy()
    # Anexa coluna 'Resíduos' (residuo observação - predito) ao df_copy
    df_copy["Resíduos"] = reg.resid

    # DEBUG prints (úteis durante desenvolvimento; em produção pode ser removido)
    print(f"Tipo de pvalues: {type(reg.pvalues)}")
    print(f"Tem index? {hasattr(reg.pvalues, 'index')}")

    # Retorna o DataFrame com resíduos e o objeto do modelo ajustado
    return df_copy, reg


def main():
    """
    Função principal que monta a interface, ajusta modelos OLS usando várias
    transformações e permite comparar abordagens (variável original vs log-transform).
    """

    # --------------------------------------------------
    # CABEÇALHO / LAYOUT DO TÍTULO
    # --------------------------------------------------
    # Cria três colunas com proporções para centralizar visualmente o título
    titulo_col_1, titulo_col_2, titulo_col_3 = st.columns([1.5, 4, 1])
    # Define o título na coluna do meio (melhor posicionamento visual)
    titulo_col_2.title("Modelo de regressão linear")

    # --------------------------------------------------
    # SELEÇÃO DA BASE DE DADOS NA BARRA LATERAL
    # --------------------------------------------------
    with st.sidebar:
        selecao_de_base = st.radio(
            label="Selecione a base de dados trabalhada",
            options=["Base de `tips`", "Base de `previsao_de_renda`"],
            horizontal=True,
        )

    # --------------------------------------------------
    # PROCESSAMENTO DA BASE DE DADOS TIPS
    # --------------------------------------------------
    if selecao_de_base == "Base de `tips`":
        # Expander explicativo (área para texto/objetivos)
        with st.expander(
            "📊 Explicação sobre objetivo da atividade e conceito da base de dados",
            expanded=True,
        ):
            st.write("---")  # separador visual dentro do expander

        # Carrega dataset de exemplo 'tips' disponibilizado pelo seaborn
        tips = sns.load_dataset("tips")

        # Cria nova feature 'net_bill': valor da conta sem a gorjeta (total_bill - tip)
        # Essa variável será usada como preditora (em diferentes transformações)
        tips["net_bill"] = tips["total_bill"] - tips["tip"]

        # --------------------------------------------------
        # CRIAÇÃO DE ABAS (TABS) NA INTERFACE
        # --------------------------------------------------
        tab_1, tab_2, tab_3 = st.tabs(
            [
                "Modelos com a variável `tip`",  # Tab 1: Modelos com variável resposta original
                "Modelos com a variável `log(tip)`",  # Tab 2: Modelos com variável resposta transformada
                "Comparação entre Tabs",  # Tab 3: Comparação entre as duas abordagens
            ]
        )

        # --------------------------------------------------
        # FEATURE ENGINEERING (transformações das features)
        # --------------------------------------------------
        tips_features_eng = tips.copy()  # cópia para trabalhar sem alterar o original

        # Tratamento para log: se existir net_bill <= 0, usar log1p (log(1+x)) para evitar -inf
        if (tips_features_eng["net_bill"] <= 0).any():
            tips_features_eng["log_net_bill"] = np.log1p(tips_features_eng["net_bill"])
        else:
            tips_features_eng["log_net_bill"] = np.log(tips_features_eng["net_bill"])

        # Cria transformação polinomial (quadrática) de net_bill
        tips_features_eng["potencia_net_bill"] = np.power(
            tips_features_eng["net_bill"], 2
        )

        # Lista com os nomes das colunas (transformações) que serão testadas nos modelos
        lista_de_ajustes = ["net_bill", "potencia_net_bill", "log_net_bill"]

        # --------------------------------------------------
        # TAB 1: Modelos usando 'tip' (variável dependente não transformada)
        # --------------------------------------------------
        with tab_1:
            # Dicionários para armazenar resultados de cada ajuste
            dfs = {}  # Armazena DataFrames com resíduos de cada modelo
            modelos = {}  # Armazena objetos dos modelos ajustados
            indices_de_performance_tab1 = (
                []
            )  # lista para guardar métricas de cada modelo

            # Ajusta um modelo para cada transformação listada em lista_de_ajustes
            for indice, col in enumerate(lista_de_ajustes):
                # Monta a fórmula incluindo variáveis categóricas + a transformação numérica
                df_modelo, modelo = criando_modelo_e_dataframe(
                    formula=f"tip ~ sex + smoker + time + {col}", df=tips_features_eng
                )

                # Chave legível para armazenar o resultado no dicionário
                key = f"ajuste_{indice}_{col}"

                # Guarda DataFrame com resíduos e o objeto do modelo
                dfs[key] = df_modelo
                modelos[key] = modelo

                # Armazena métricas relevantes para comparação futura
                indices_de_performance_tab1.append(
                    {
                        "Modelo": f"Modelo_{col}",
                        "R²": modelo.rsquared,  # Coeficiente de determinação
                        "R² Ajustado": modelo.rsquared_adj,  # R² ajustado pelo número de variáveis
                        "AIC": modelo.aic,  # Critério de Informação de Akaike
                        "BIC": modelo.bic,  # Critério de Informação Bayesiano
                        "F-Statistic": modelo.fvalue,  # Estatística F do modelo
                        "Prob (F-Statistic)": modelo.f_pvalue,  # p-valor da estatística F
                        "Número de observações": modelo.nobs,  # Número de observações
                    }
                )

            # Opções (rótulos) exibidas ao usuário para escolher qual ajuste visualizar
            opcoes_modelos = {
                "Modelo Linear Simples (net_bill)": "ajuste_0_net_bill",
                "Modelo Quadrático (potencia_net_bill)": "ajuste_1_potencia_net_bill",
                "Modelo Logarítmico (log_net_bill)": "ajuste_2_log_net_bill",
            }

            # Radio widget para selecionar modelo a visualizar
            selecao_modelo = st.radio(
                options=list(opcoes_modelos.keys()),
                label="Selecione o modelo que deseja visualizar os resíduos",
                horizontal=True,
            )

            # Recupera a chave técnica correspondente à seleção do usuário
            chave_tecnica = opcoes_modelos[selecao_modelo]

            # Cria figura e eixo matplotlib para desenhar gráfico de resíduos
            fig, ax = plt.subplots()

            # Mapeamento para recuperar o nome da feature correspondente à chave técnica
            mapeamento_features = {
                "ajuste_0_net_bill": "net_bill",
                "ajuste_1_potencia_net_bill": "potencia_net_bill",
                "ajuste_2_log_net_bill": "log_net_bill",
            }

            # Nome da variável a ser usada no eixo x do gráfico
            feature_name = mapeamento_features[chave_tecnica]

            # Plota scatterplot: feature transformada (x) vs resíduos do modelo (y)
            sns.scatterplot(
                x=tips_features_eng[feature_name],  # valores da feature
                y=dfs[chave_tecnica]["Resíduos"],  # resíduos do modelo
                ax=ax,
            )

            # Ajustes visuais do gráfico
            plt.tight_layout()
            plt.title(f"Gráfico de Resíduos - {selecao_modelo}")
            plt.xlabel(f"Variável: {feature_name}")
            plt.ylabel("Resíduos")
            plt.axhline(0, color="black", linestyle="--")  # linha de referência em zero

            # Salva a figura em memória e exibe no Streamlit
            buff = BytesIO()
            fig.savefig(buff, format="png", bbox_inches="tight")
            buff.seek(0)

            # Layout com duas colunas: imagem à esquerda, controles/métricas à direita
            col1, col2 = st.columns(2)
            with col1:
                st.image(buff)  # Exibe o gráfico de resíduos

            with col2:
                # Checkbox para exibir o summary completo do modelo
                if st.checkbox("Exibir sumário do modelo", key="checkbox_tab1"):
                    st.write("##### Sumário do modelo")
                    st.code(str(modelos[chave_tecnica].summary()), language="text")

                # Trabalha com p-values para determinar significância das variáveis
                modelo = modelos[chave_tecnica]
                p_values = modelo.pvalues
                # DataFrame com nomes das variáveis e seus p-values
                df_pvalues = pd.DataFrame(
                    {"Variáveis": p_values.index, "P-value": p_values.values}
                )

                st.info(
                    "Somente consideramos relevantes aquelas variáveis que apresentarem um p-value inferior a 5%"
                )

                # Separa variáveis significativas (<0.05) e não-significativas (>=0.05)
                nao_significativas = p_values[p_values >= 0.05]

                # Se existir ao menos uma variável não significativa, mostra erro/aviso
                if len(nao_significativas) > 0:
                    st.error(
                        f"É necessário deletar as variáveis {df_pvalues.loc[df_pvalues['P-value'] >= 0.05, 'Variáveis'].to_list()}"
                    )
                else:
                    st.success("É aconselhável manter todas as variáveis")

                # Botão para exibir análise comparativa dos modelos
                informe_modelo = st.button(
                    "Informe o melhor modelo",
                    help="Baseado em indicadores como AIC, R² e R² Ajustado",
                    key="botao_tab1",
                )

                if informe_modelo:
                    # Converte a lista de métricas em DataFrame e exibe formatado
                    df_comparacao = pd.DataFrame(indices_de_performance_tab1)
                    st.write("#### Métricas de Comparação - Tab 1")
                    st.dataframe(
                        df_comparacao.style.format(
                            {
                                "R²": "{:.4f}",
                                "R² Ajustado": "{:.4f}",
                                "AIC": "{:.2f}",
                                "BIC": "{:.2f}",
                                "F-Statistic": "{:.2f}",
                                "Prob (F-statistic)": "{:.4f}",
                            }
                        )
                        .highlight_max(subset=["R²", "R² Ajustado"], color="lightgreen")
                        .highlight_min(subset=["AIC", "BIC"], color="lightgreen"),
                        hide_index=True,
                    )

                    # --------------------------------------------------
                    # DETERMINAR "MELHORES" MODELOS POR DIFERENTES CRITÉRIOS (Tab 1)
                    # --------------------------------------------------
                    melhor_r2 = df_comparacao.loc[df_comparacao["R²"].idxmax()]
                    melhor_r2_ajustado = df_comparacao.loc[
                        df_comparacao["R² Ajustado"].idxmax()
                    ]
                    menor_aic = df_comparacao.loc[df_comparacao["AIC"].idxmin()]
                    menor_bic = df_comparacao.loc[df_comparacao["BIC"].idxmin()]

                    # Exibe os "melhores" por critério em métricas
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("#### 🏆 Melhor Modelo por Critério")
                        st.metric(
                            label="**Melhor R²**",
                            value=melhor_r2["Modelo"],
                            delta=f"R²: {melhor_r2['R²']:.4f}",
                        )
                        st.metric(
                            label="**Melhor R² Ajustado**",
                            value=melhor_r2_ajustado["Modelo"],
                            delta=f"R² Ajustado: {melhor_r2_ajustado['R² Ajustado']:.4f}",
                        )

                    with col2:
                        st.write("#### 📊 Outras Métricas")
                        st.metric(
                            label="**Menor AIC**",
                            value=menor_aic["Modelo"],
                            delta=f"AIC: {menor_aic['AIC']:.2f}",
                        )
                        st.metric(
                            label="**Menor BIC**",
                            value=menor_bic["Modelo"],
                            delta=f"BIC: {menor_bic['BIC']:.2f}",
                        )

                    # Reúne os "melhores" de cada critério para decidir recomendação
                    st.write("#### 💡 Análise e Recomendação")
                    modelos_melhores = set(
                        [
                            melhor_r2["Modelo"],
                            melhor_r2_ajustado["Modelo"],
                            menor_aic["Modelo"],
                            menor_bic["Modelo"],
                        ]
                    )

                    # Se todas as métricas apontam para o mesmo modelo, recomenda diretamente
                    if len(modelos_melhores) == 1:
                        modelo_recomendado = list(modelos_melhores)[0]
                        st.success(f"**Modelo Recomendado:** {modelo_recomendado}")
                    else:
                        # Caso de divergência entre métricas: prioriza R² Ajustado como fallback
                        st.warning(
                            "**Análise:** As métricas apontam para modelos diferentes."
                        )
                        st.info(
                            """
                        **Interpretação:**
                        - **R²**: Explica a variância dos dados (quanto maior, melhor)
                        - **R² Ajustado**: R² penalizado pelo número de variáveis
                        - **AIC/BIC**: Critérios de informação (quanto menor, melhor)
                        - Em caso de divergência, priorize R² Ajustado para comparação justa
                        """
                        )
                        modelo_recomendado = melhor_r2_ajustado["Modelo"]
                        st.success(
                            f"**Modelo Recomendado (baseado no R² Ajustado):** {modelo_recomendado}"
                        )

                    # Guarda a recomendação na sessão para uso posterior (aba 3)
                    st.session_state.modelo_recomendado_tab_1 = modelo_recomendado

            # Fecha a figura para liberar memória (boa prática)
            plt.close()

        # --------------------------------------------------
        # TAB 2: Modelos usando log(tip) como variável dependente
        # --------------------------------------------------
        with tab_2:
            # Estruturas similares à Tab 1, mas para modelos com log(tip)
            dfs_tab_two = {}
            modelos_tab_two = {}
            indices_de_performance_tab2 = []

            # Ajusta modelos similares aos da Tab 1, mas com np.log(tip) como resposta
            for indice, col in enumerate(lista_de_ajustes):
                df_modelo, modelo = criando_modelo_e_dataframe(
                    formula=f"np.log(tip) ~ sex + smoker + time + {col}",
                    df=tips_features_eng,
                )
                key = f"ajuste_log_{indice}_{col}"
                dfs_tab_two[key] = df_modelo
                modelos_tab_two[key] = modelo

                indices_de_performance_tab2.append(
                    {
                        "Modelo": f"Modelo_log_{col}",
                        "R²": modelo.rsquared,
                        "R² Ajustado": modelo.rsquared_adj,
                        "AIC": modelo.aic,
                        "BIC": modelo.bic,
                        "F-Statistic": modelo.fvalue,
                        "Prob (F-Statistic)": modelo.f_pvalue,
                        "Número de observações": modelo.nobs,
                    }
                )

            # Opções para o radio do Tab 2
            opcoes_modelos_tab_two = {
                "Modelo Linear Simples (net_bill)": "ajuste_log_0_net_bill",
                "Modelo Quadrático (potencia_net_bill)": "ajuste_log_1_potencia_net_bill",
                "Modelo Logarítmico (log_net_bill)": "ajuste_log_2_log_net_bill",
            }

            selecao_modelo_tab_two = st.radio(
                options=list(opcoes_modelos_tab_two.keys()),
                label="Selecione o modelo que deseja visualizar os resíduos",
                horizontal=True,
                key="selecao_de_modelo_tab_two_key",
            )
            chave_tecnica_tab_two = opcoes_modelos_tab_two[selecao_modelo_tab_two]

            # Plota resíduos para a escolha do usuário na Tab 2
            fig, ax = plt.subplots()
            mapeamento_features_tab_two = {
                "ajuste_log_0_net_bill": "net_bill",
                "ajuste_log_1_potencia_net_bill": "potencia_net_bill",
                "ajuste_log_2_log_net_bill": "log_net_bill",
            }
            feature_name_tab_two = mapeamento_features_tab_two[chave_tecnica_tab_two]

            sns.scatterplot(
                x=tips_features_eng[feature_name_tab_two],
                y=dfs_tab_two[chave_tecnica_tab_two]["Resíduos"],
                ax=ax,
            )

            plt.tight_layout()
            plt.title(f"Gráfico de Resíduos - {selecao_modelo_tab_two}")
            plt.xlabel(f"Variável: {feature_name_tab_two}")
            plt.ylabel("Resíduos")
            plt.axhline(0, color="black", linestyle="--")

            buff = BytesIO()
            fig.savefig(buff, format="png", bbox_inches="tight")
            buff.seek(0)
            col1, col2 = st.columns(2)
            with col1:
                st.image(buff)
            with col2:
                # Controles similares aos da Tab 1
                if st.checkbox("Exibir sumário do modelo", key="checkbox_tab2"):
                    st.write("##### Sumário do modelo")
                    st.code(
                        str(modelos_tab_two[chave_tecnica_tab_two].summary()),
                        language="text",
                    )

                modelo_tab_two = modelos_tab_two[chave_tecnica_tab_two]
                p_values_tab_two = modelo_tab_two.pvalues
                df_pvalues_tab_two = pd.DataFrame(
                    {
                        "Variáveis": p_values_tab_two.index,
                        "P-value": p_values_tab_two.values,
                    }
                )

                st.info(
                    "Somente consideramos relevantes aquelas variáveis que apresentarem um p-value inferior a 5%"
                )

                nao_significativas_tab_two = p_values_tab_two[p_values_tab_two >= 0.05]
                if len(nao_significativas_tab_two) > 0:
                    st.error(
                        f"É necessário deletar as variáveis {df_pvalues_tab_two.loc[df_pvalues_tab_two['P-value'] >= 0.05, 'Variáveis'].to_list()}"
                    )
                else:
                    st.success("É aconselhável manter todas as variáveis")

                # Botão que exibe as métricas comparativas para Tab 2
                informe_modelo_tab2 = st.button(
                    "Informe o melhor modelo",
                    help="Baseado em indicadores como AIC, R² e R² Ajustado",
                    key="botao_tab2",
                )

                if informe_modelo_tab2:
                    df_comparacao_tab2 = pd.DataFrame(indices_de_performance_tab2)
                    st.write("#### Métricas de Comparação - Tab 2")
                    st.dataframe(
                        df_comparacao_tab2.style.format(
                            {
                                "R²": "{:.4f}",
                                "R² Ajustado": "{:.4f}",
                                "AIC": "{:.2f}",
                                "BIC": "{:.2f}",
                                "F-Statistic": "{:.2f}",
                                "Prob (F-statistic)": "{:.4f}",
                            }
                        )
                        .highlight_max(subset=["R²", "R² Ajustado"], color="lightgreen")
                        .highlight_min(subset=["AIC", "BIC"], color="lightgreen"),
                        hide_index=True,
                    )

                    # Determina melhores modelos por critério na Tab 2
                    melhor_r2_tab2 = df_comparacao_tab2.loc[
                        df_comparacao_tab2["R²"].idxmax()
                    ]
                    melhor_r2_ajustado_tab2 = df_comparacao_tab2.loc[
                        df_comparacao_tab2["R² Ajustado"].idxmax()
                    ]
                    menor_aic_tab2 = df_comparacao_tab2.loc[
                        df_comparacao_tab2["AIC"].idxmin()
                    ]
                    menor_bic_tab2 = df_comparacao_tab2.loc[
                        df_comparacao_tab2["BIC"].idxmin()
                    ]

                    # Exibição das métricas "vencedoras"
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("#### 🏆 Melhor Modelo por Critério")
                        st.metric(
                            label="**Melhor R²**",
                            value=melhor_r2_tab2["Modelo"],
                            delta=f"R²: {melhor_r2_tab2['R²']:.4f}",
                        )
                        st.metric(
                            label="**Melhor R² Ajustado**",
                            value=melhor_r2_ajustado_tab2["Modelo"],
                            delta=f"R² Ajustado: {melhor_r2_ajustado_tab2['R² Ajustado']:.4f}",
                        )
                    with col2:
                        st.write("#### 📊 Outras Métricas")
                        st.metric(
                            label="**Menor AIC**",
                            value=menor_aic_tab2["Modelo"],
                            delta=f"AIC: {menor_aic_tab2['AIC']:.2f}",
                        )
                        st.metric(
                            label="**Menor BIC**",
                            value=menor_bic_tab2["Modelo"],
                            delta=f"BIC: {menor_bic_tab2['BIC']:.2f}",
                        )

                    st.write("#### 💡 Análise e Recomendação")

                    modelos_melhores_tab2 = set(
                        [
                            melhor_r2_tab2["Modelo"],
                            melhor_r2_ajustado_tab2["Modelo"],
                            menor_aic_tab2["Modelo"],
                            menor_bic_tab2["Modelo"],
                        ]
                    )

                    if len(modelos_melhores_tab2) == 1:
                        modelo_recomendado_tab2 = list(modelos_melhores_tab2)[0]
                        st.success(f"**Modelo Recomendado:** {modelo_recomendado_tab2}")
                    else:
                        st.warning(
                            "**Análise:** As métricas apontam para modelos diferentes."
                        )
                        st.info(
                            """
                        **Interpretação:**
                        - **R²**: Explica a variância dos dados (quanto maior, melhor)
                        - **R² Ajustado**: R² penalizado pelo número de variáveis
                        - **AIC/BIC**: Critérios de informação (quanto menor, melhor)
                        - Em caso de divergência, priorize R² Ajustado para comparação justa
                        """
                        )

                        modelo_recomendado_tab2 = melhor_r2_ajustado_tab2["Modelo"]
                        st.success(
                            f"**Modelo Recomendado (baseado no R² Ajustado):** {modelo_recomendado_tab2}"
                        )

                    # Armazena recomendação da Tab 2 na sessão
                    st.session_state.modelo_recomendado_tab_2 = modelo_recomendado_tab2

            plt.close()

        # --------------------------------------------------
        # TAB 3: Comparação entre as recomendações das Tabs 1 e 2
        # --------------------------------------------------
        with tab_3:
            st.header("🔍 Comparação entre Modelos das Duas Tabs")

            comparar_modelos = st.button(
                "Comparar Modelos entre Tabs",
                help="Compare os melhores modelos de cada tab para determinar qual abordagem é mais eficaz",
                key="botao_comparacao_tabs",
            )

            if comparar_modelos:
                # Verifica se ambas as tabs já produziram recomendações
                if (
                    st.session_state.modelo_recomendado_tab_1 is None
                    or st.session_state.modelo_recomendado_tab_2 is None
                ):
                    st.warning(
                        "⚠️ Por favor, execute primeiro a análise de melhores modelos em ambas as tabs antes de comparar."
                    )
                else:
                    st.write("### 📊 Comparação entre Abordagens")

                    # Junta métricas apenas dos modelos recomendados por cada tab
                    todas_metricas = []

                    # Adiciona métricas do Modelo recomendado da Tab 1
                    for metrica in indices_de_performance_tab1:
                        if (
                            metrica["Modelo"]
                            == st.session_state.modelo_recomendado_tab_1
                        ):
                            metrica_copy = metrica.copy()
                            metrica_copy["Abordagem"] = "Variável Original (tip)"
                            todas_metricas.append(metrica_copy)

                    # Adiciona métricas do Modelo recomendado da Tab 2
                    for metrica in indices_de_performance_tab2:
                        if (
                            metrica["Modelo"]
                            == st.session_state.modelo_recomendado_tab_2
                        ):
                            metrica_copy = metrica.copy()
                            metrica_copy["Abordagem"] = "Transformação Log (log(tip))"
                            todas_metricas.append(metrica_copy)

                    # DataFrame final com métricas lado a lado
                    df_comparacao_final = pd.DataFrame(todas_metricas)

                    # Exibe tabela comparativa formatada
                    st.write("#### 📈 Métricas dos Melhores Modelos por Abordagem")
                    st.dataframe(
                        df_comparacao_final.style.format(
                            {
                                "R²": "{:.4f}",
                                "R² Ajustado": "{:.4f}",
                                "AIC": "{:.2f}",
                                "BIC": "{:.2f}",
                                "F-Statistic": "{:.2f}",
                                "Prob (F-statistic)": "{:.4f}",
                            }
                        )
                        .highlight_max(subset=["R²", "R² Ajustado"], color="lightgreen")
                        .highlight_min(subset=["AIC", "BIC"], color="lightgreen"),
                        hide_index=True,
                    )

                    # --------------------------------------------------
                    # RECOMENDAÇÃO FINAL - compara R² Ajustado e AIC
                    # --------------------------------------------------
                    st.write("#### 🏆 Recomendação Final")

                    # Extrai R² Ajustado e AIC das duas abordagens
                    r2_ajustado_tab1 = df_comparacao_final[
                        df_comparacao_final["Abordagem"] == "Variável Original (tip)"
                    ]["R² Ajustado"].values[0]
                    r2_ajustado_tab2 = df_comparacao_final[
                        df_comparacao_final["Abordagem"]
                        == "Transformação Log (log(tip))"
                    ]["R² Ajustado"].values[0]

                    aic_tab1 = df_comparacao_final[
                        df_comparacao_final["Abordagem"] == "Variável Original (tip)"
                    ]["AIC"].values[0]
                    aic_tab2 = df_comparacao_final[
                        df_comparacao_final["Abordagem"]
                        == "Transformação Log (log(tip))"
                    ]["AIC"].values[0]

                    # Mostra métricas comparadas via st.metric com delta
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="**Melhor R² Ajustado**",
                            value=(
                                "Transformação Log"
                                if r2_ajustado_tab2 > r2_ajustado_tab1
                                else "Variável Original"
                            ),
                            delta=f"Diferença: {abs(r2_ajustado_tab2 - r2_ajustado_tab1):.4f}",
                            delta_color=(
                                "normal"
                                if r2_ajustado_tab2 > r2_ajustado_tab1
                                else "inverse"
                            ),
                        )

                    with col2:
                        st.metric(
                            label="**Menor AIC**",
                            value=(
                                "Transformação Log"
                                if aic_tab2 < aic_tab1
                                else "Variável Original"
                            ),
                            delta=f"Diferença: {abs(aic_tab2 - aic_tab1):.2f}",
                            delta_color="normal" if aic_tab2 < aic_tab1 else "inverse",
                        )

                    # Conta critérios favoráveis para decidir recomendação final
                    criterios_favor_tab2 = 0
                    criterios_favor_tab1 = 0

                    if r2_ajustado_tab2 > r2_ajustado_tab1:
                        criterios_favor_tab2 += 1
                    else:
                        criterios_favor_tab1 += 1

                    if aic_tab2 < aic_tab1:
                        criterios_favor_tab2 += 1
                    else:
                        criterios_favor_tab1 += 1

                    # Emite recomendação final baseada na contagem
                    if criterios_favor_tab2 > criterios_favor_tab1:
                        st.success(
                            "🎯 **Recomendação Final:** Use a abordagem com Transformação Log (log(tip))"
                        )
                        st.info(
                            """
                        **Justificativa:**
                        - Melhor ajuste aos dados (maior R² Ajustado)
                        - Menor complexidade do modelo (menor AIC)
                        - Transformação logarítmica pode ajudar com heterocedasticidade
                        """
                        )
                    elif criterios_favor_tab1 > criterios_favor_tab2:
                        st.success(
                            "🎯 **Recomendação Final:** Use a abordagem com Variável Original (tip)"
                        )
                        st.info(
                            """
                        **Justificativa:**
                        - Melhor ajuste aos dados (maior R² Ajustado)
                        - Menor complexidade do modelo (menor AIC)
                        - Transformação logarítmica pode ajudar com heterocedasticidade
                        """
                        )
                    elif criterios_favor_tab1 > criterios_favor_tab2:
                        st.success(
                            "🎯 **Recomendação Final:** Use a abordagem com Variável Original (tip)"
                        )
                        st.info(
                            """
                        **Justificativa:**
                        - Melhor ajuste aos dados (maior R² Ajustado)
                        - Menor complexidade do modelo (menor AIC)
                        - Interpretação mais direta dos coeficientes
                        """
                        )
                    else:
                        st.warning(
                            "⚖️ **Empate Técnico:** Ambas as abordagens têm méritos similares"
                        )
                        st.info(
                            """
                        **Considerações:**
                        - Avalie a distribuição dos resíduos
                        - Considere a interpretabilidade dos resultados
                        - Verifique suposições do modelo linear
                        """
                        )

    # --------------------------------------------------
    # PROCESSAMENTO DA BASE DE DADOS PREVISÃO DE RENDA
    # --------------------------------------------------
    if selecao_de_base == "Base de `previsao_de_renda`":
        # Interface para upload do arquivo de previsão de renda
        with st.sidebar:
            uploaded_file = st.file_uploader(
                "Faça o upload da base `previsao_renda.csv`", type=["csv"]
            )

        # Processamento do arquivo uploadado
        if (uploaded_file is not None) & (
            selecao_de_base == "Base de `previsao_de_renda`"
        ):
            # Carrega o dataset e remove valores missing
            previsao_renda = pd.read_csv(uploaded_file)
            previsao_renda.dropna(inplace=True)

            # Inicializar no session_state se não existir
            if "previsao_renda_processed" not in st.session_state:
                st.session_state.previsao_renda_processed = previsao_renda.copy()

            # Usar dados do session_state (permite persistência entre interações)
            X = st.session_state.previsao_renda_processed.drop(columns=["renda"])

            # Widget para seleção de variáveis a serem removidas do modelo
            variaveis_para_remover = st.multiselect(
                label="Selecione as variáveis da base `previsao_renda`",
                options=X.columns,
            )

            # Layout com botões de controle
            col1, col2 = st.columns([1, 4])
            with col1:
                # Botão para remover variáveis selecionadas
                if st.button("Remover variáveis selecionadas"):
                    if variaveis_para_remover:
                        st.session_state.previsao_renda_processed = (
                            st.session_state.previsao_renda_processed.drop(
                                columns=variaveis_para_remover
                            )
                        )
                        st.success(
                            f"✅ Variáveis removidas: {', '.join(variaveis_para_remover)}"
                        )
                        st.rerun()  # Recarregar a página para atualizar as opções
                    else:
                        st.warning("⚠️ Selecione pelo menos uma variável para remover")

            with col2:
                # Botão para resetar aos dados originais
                if st.button("Resetar para dados originais"):
                    st.session_state.previsao_renda_processed = previsao_renda.copy()
                    st.success("✅ Dados resetados para o original")
                    st.rerun()

            # Atualizar X após possíveis remoções
            X = st.session_state.previsao_renda_processed.drop(columns=["renda"])
            variaveis_explicativas = X.columns.to_list()

            # Verifica se ainda há variáveis para modelar
            if variaveis_explicativas:
                # Constrói fórmula dinamicamente com as variáveis restantes
                formula = "renda ~ " + " + ".join(variaveis_explicativas)

                # Ajusta o modelo de regressão
                df, reg_previsao_renda = criando_modelo_e_dataframe(
                    formula=formula, df=st.session_state.previsao_renda_processed
                )

                # Exibe resultados básicos do modelo
                st.write("### Modelo de Previsão de Renda")
                st.write(f"**Fórmula usada:** `{formula}`")
                st.write(f"**R² do modelo:** {reg_previsao_renda.rsquared:.4f}")
                st.write(f"**Número de observações:** {reg_previsao_renda.nobs}")
                st.write(f"**Número de variáveis:** {len(variaveis_explicativas)}")

                # Cria abas para análise detalhada
                var_pvalue_tab, res_distribuition_tab = st.tabs(
                    [
                        "Visualize a performance das variáveis",
                        "Verifique a distribuição dos resíduos",
                    ]
                )

                # --------------------------------------------------
                # ABA: ANÁLISE DE SIGNIFICÂNCIA DAS VARIÁVEIS
                # --------------------------------------------------
                with var_pvalue_tab:
                    # Análise de significância das variáveis através de p-values
                    st.write("#### 📊 Significância das Variáveis (p-values)")
                    p_values = reg_previsao_renda.pvalues
                    df_pvalues = pd.DataFrame(
                        {"Variável": p_values.index, "P-value": p_values.values}
                    )

                    # Destacar variáveis não significativas (p-value >= 0.05)
                    st.dataframe(
                        df_pvalues.style.format({"P-value": "{:.4f}"})
                        .highlight_between(
                            subset=["P-value"], left=0.05, right=1.0, color="lightcoral"
                        )
                        .highlight_between(
                            subset=["P-value"], left=0.0, right=0.05, color="lightgreen"
                        ),
                        hide_index=True,
                    )

                # --------------------------------------------------
                # ABA: ANÁLISE DE DISTRIBUIÇÃO DOS RESÍDUOS
                # --------------------------------------------------
                with res_distribuition_tab:
                    # Opção para exibir sumário completo do modelo
                    if st.checkbox("Exibir sumário do modelo de renda"):
                        st.code(str(reg_previsao_renda.summary()), language="text")

                    # Adicionar análise de resíduos com visualizações interativas
                    st.write("### 📈 Análise de Resíduos")

                    # Calcula resíduos e valores preditos
                    df["residuos"] = reg_previsao_renda.resid
                    df["predito"] = reg_previsao_renda.fittedvalues

                    # Selectbox para escolher variável para análise detalhada
                    variavel_selecionada = st.selectbox(
                        "Selecione uma variável para análise dos resíduos:",
                        options=variaveis_explicativas,
                        key="resid_analysis",
                    )

                    # Análise visual para a variável selecionada
                    if variavel_selecionada:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(
                                f"#### Boxplot - {variavel_selecionada} vs Resíduos"
                            )

                            # Verificar se a variável é numérica ou categórica
                            if (
                                df[variavel_selecionada].dtype in ["int64", "float64"]
                                and df[variavel_selecionada].nunique() > 10
                            ):
                                # Para variáveis numéricas com muitos valores, criar categorias
                                df_temp = df.copy()
                                df_temp[f"{variavel_selecionada}_cat"] = pd.cut(
                                    df_temp[variavel_selecionada],
                                    bins=5,
                                    duplicates="drop",
                                )

                                # Converter intervalos para strings para evitar erro de serialização
                                df_temp[f"{variavel_selecionada}_cat_str"] = df_temp[
                                    f"{variavel_selecionada}_cat"
                                ].astype(str)

                                fig_box = px.box(
                                    df_temp,
                                    x=f"{variavel_selecionada}_cat_str",
                                    y="residuos",
                                    title=f"Distribuição dos Resíduos por {variavel_selecionada}",
                                    labels={
                                        f"{variavel_selecionada}_cat_str": variavel_selecionada
                                    },
                                )
                            else:
                                # Para variáveis categóricas ou numéricas com poucos valores
                                df_temp = df.copy()
                                df_temp[f"{variavel_selecionada}_str"] = df_temp[
                                    variavel_selecionada
                                ].astype(str)

                                fig_box = px.box(
                                    df_temp,
                                    x=f"{variavel_selecionada}_str",
                                    y="residuos",
                                    title=f"Distribuição dos Resíduos por {variavel_selecionada}",
                                    labels={
                                        f"{variavel_selecionada}_str": variavel_selecionada
                                    },
                                )

                            fig_box.update_layout(
                                xaxis_tickangle=-45,
                                xaxis_title=variavel_selecionada,
                                yaxis_title="Resíduos",
                            )
                            st.plotly_chart(fig_box, use_container_width=True)

                        with col2:
                            st.write(
                                f"#### Scatter Plot - {variavel_selecionada} vs Resíduos"
                            )

                            if df[variavel_selecionada].dtype in ["int64", "float64"]:
                                # Para variáveis numéricas: scatter plot com tendência
                                fig_scatter = px.scatter(
                                    df,
                                    x=variavel_selecionada,
                                    y="residuos",
                                    title=f"Resíduos vs {variavel_selecionada}",
                                    trendline="lowess",  # LOWESS: Locally Weighted Scatterplot Smoothing
                                    opacity=0.6,
                                )

                                # Adicionar linha horizontal em y=0 para referência
                                fig_scatter.add_hline(
                                    y=0, line_dash="dash", line_color="red"
                                )
                                fig_scatter.update_layout(
                                    xaxis_title=variavel_selecionada,
                                    yaxis_title="Resíduos",
                                )

                            else:
                                # Para variáveis categóricas: violin plot
                                df_temp = df.copy()
                                df_temp[f"{variavel_selecionada}_str"] = df_temp[
                                    variavel_selecionada
                                ].astype(str)

                                fig_scatter = px.violin(
                                    df_temp,
                                    x=f"{variavel_selecionada}_str",
                                    y="residuos",
                                    title=f"Distribuição dos Resíduos por {variavel_selecionada}",
                                    box=True,  # Inclui boxplot dentro do violin
                                )

                                fig_scatter.add_hline(
                                    y=0, line_dash="dash", line_color="red"
                                )
                                fig_scatter.update_layout(
                                    xaxis_title=variavel_selecionada,
                                    yaxis_title="Resíduos",
                                )

                            st.plotly_chart(fig_scatter, use_container_width=True)

                    # --------------------------------------------------
                    # DIAGNÓSTICO AVANÇADO DO MODELO
                    # --------------------------------------------------
                    st.write("### 🔍 Diagnóstico do Modelo")

                    col3, col4 = st.columns(2)

                    with col3:
                        # Gráfico de resíduos vs valores preditos
                        st.write("#### Resíduos vs Valores Preditos")
                        fig_resid_fitted = px.scatter(
                            df,
                            x="predito",
                            y="residuos",
                            title="Resíduos vs Valores Preditos",
                            trendline="lowess",
                            opacity=0.6,
                        )
                        fig_resid_fitted.add_hline(
                            y=0, line_dash="dash", line_color="red"
                        )
                        fig_resid_fitted.update_layout(
                            xaxis_title="Valores Preditos", yaxis_title="Resíduos"
                        )
                        st.plotly_chart(fig_resid_fitted, use_container_width=True)

                    with col4:
                        # QQ-Plot para verificar normalidade dos resíduos
                        st.write("#### QQ-Plot dos Resíduos")

                        # Calcular QQ-plot manualmente
                        residuos_std = (df["residuos"] - df["residuos"].mean()) / df[
                            "residuos"
                        ].std()
                        theoretical_quantiles = stats.norm.ppf(
                            np.linspace(0.01, 0.99, len(residuos_std))
                        )

                        qq_data = pd.DataFrame(
                            {
                                "Theoretical Quantiles": theoretical_quantiles,
                                "Sample Quantiles": np.sort(residuos_std),
                            }
                        )

                        fig_qq = px.scatter(
                            qq_data,
                            x="Theoretical Quantiles",
                            y="Sample Quantiles",
                            title="QQ-Plot - Normalidade dos Resíduos",
                            opacity=0.6,
                        )

                        # Adicionar linha de referência (45 graus) para normalidade perfeita
                        min_val = min(
                            qq_data["Theoretical Quantiles"].min(),
                            qq_data["Sample Quantiles"].min(),
                        )
                        max_val = max(
                            qq_data["Theoretical Quantiles"].max(),
                            qq_data["Sample Quantiles"].max(),
                        )

                        # Criar linha de referência separadamente
                        line_trace = go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode="lines",
                            line=dict(dash="dash", color="red"),
                            name="Linha de Referência",
                        )

                        fig_qq.add_trace(line_trace)
                        fig_qq.update_layout(
                            xaxis_title="Quantis Teóricos",
                            yaxis_title="Quantis Amostrais",
                        )

                        st.plotly_chart(fig_qq, use_container_width=True)

                    # --------------------------------------------------
                    # ESTATÍSTICAS DESCRITIVAS DOS RESÍDUOS
                    # --------------------------------------------------
                    st.write("### 📊 Estatísticas dos Resíduos")
                    col5, col6, col7, col8 = st.columns(4)

                    with col5:
                        st.metric("Média dos Resíduos", f"{df['residuos'].mean():.4f}")
                    with col6:
                        st.metric("Desvio Padrão", f"{df['residuos'].std():.4f}")
                    with col7:
                        st.metric("Assimetria", f"{df['residuos'].skew():.4f}")
                    with col8:
                        st.metric("Curtose", f"{df['residuos'].kurtosis():.4f}")


# ------------------------------------------------------
# PONTO DE ENTRADA DA APLICAÇÃO
# ------------------------------------------------------
if __name__ == "__main__":
    main()  # Executa a função principal ao rodar o script
