import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import BytesIO

import statsmodels.formula.api as smf


# ------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ------------------------------------------------------
st.set_page_config(
    page_title="Trabalhando com modelos de regressão Linear",  # Título exibido na aba do navegador
    layout="wide",  # Usa largura total da janela
    initial_sidebar_state="expanded",  # sidebar aberta por padrão
    page_icon="varig_icon.png",  # ícone da aplicação (arquivo esperado localmente)
)


def main():
    # --------------------------------------------------
    # CABEÇALHO / LAYOUT DO TÍTULO
    # --------------------------------------------------
    # Cria três colunas com proporções diferentes para posicionar o título centralizado
    titulo_col_1, titulo_col_2, titulo_col_3 = st.columns([1.5, 4, 1])
    # Define o título na coluna do meio — isso evita que o título fique muito à esquerda
    titulo_col_2.title("Modelo de regressão linear baseado no dataset `tips`")

    # Expander explicativo — espaço para colocar descrição/objetivo da atividade
    # (aqui está vazio, mas reserva um lugar para explicações)
    with st.expander(
        "📊 Explicação sobre objetivo da atividade e conceito da base de dados",
        expanded=True,
    ):
        st.write("---")  # separador visual simples dentro do expander

    # --------------------------------------------------
    # CARREGAMENTO E PREPARAÇÃO DOS DADOS (DATASET `tips`)
    # --------------------------------------------------
    # Carrega o dataset 'tips' diretamente do seaborn (dataset de exemplo)
    tips = sns.load_dataset("tips")

    # Cria variável 'tip_pct' que representa a proporção da gorjeta
    # Formula: tip / (total_bill - tip) -> proporção relativa ao valor sem tip
    tips["tip_pct"] = tips["tip"] / (tips["total_bill"] - tips["tip"])

    # Cria variável 'net_bill' que corresponde ao valor líquido da conta (sem a gorjeta)
    tips["net_bill"] = tips["total_bill"] - tips["tip"]

    # --------------------------------------------------
    # AJUSTE DE MODELO LINEAR SIMPLES (BASE)
    # --------------------------------------------------
    # Ajusta um modelo de regressão linear (OLS) usando statsmodels
    # Equação: tip_pct ~ net_bill (tip_pct explicado por net_bill)
    model_regr_tip_pct = smf.ols(f"tip_pct~net_bill", data=tips).fit()

    # --------------------------------------------------
    # BLOCO DE MODELOS ALTERNATIVOS (TRANSFORMAÇÕES)
    # --------------------------------------------------
    # Expander que contém diversos modelos com transformações nas variáveis
    with st.expander("Conjunto de modelos de regressão linear", expanded=False):
        # Cria cópias do dataframe original para armazenar resíduos de cada modelo
        df_modelo_original = tips.copy()
        df_modelo_01 = tips.copy()
        df_modelo_02 = tips.copy()
        df_modelo_03 = tips.copy()
        df_modelo_04 = tips.copy()

        # Reajusta o modelo base (poderia reutilizar model_regr_tip_pct acima — aqui é redundante)
        model_regr_tip_pct = smf.ols(f"tip_pct~net_bill", data=tips).fit()

        # Modelo 1: logaritmo de net_bill como preditor
        model_01 = smf.ols("tip_pct ~ np.log(net_bill)", data=tips).fit()

        # Modelo 2: transformação polinomial (net_bill ao quadrado)
        model_02 = smf.ols("tip_pct ~ np.power(net_bill,2)", data=tips).fit()

        # Modelo 3: logaritmo da variável dependente tip_pct (transformação na resposta)
        model_03 = smf.ols("np.log(tip_pct) ~ net_bill", data=tips).fit()

        # Modelo 4: log em ambas as variáveis (resposta e preditor)
        model_04 = smf.ols("np.log(tip_pct) ~ np.log(net_bill)", data=tips).fit()

        # Calcula e anexa os resíduos de cada modelo nos respectivos dataframes-cópia
        # 'resíduos' = y_observado - y_predito (modelo). Esses serão usados para diagnóstico gráfico.
        df_modelo_original["resíduos"] = model_regr_tip_pct.resid
        df_modelo_01["resíduos"] = model_01.resid
        df_modelo_02["resíduos"] = model_02.resid
        df_modelo_03["resíduos"] = model_03.resid
        df_modelo_04["resíduos"] = model_04.resid

        # Lista legível com as opções de modelo para o usuário escolher via radio button
        selecao_de_modelos_options = [
            "Modelo original",
            "Modelo com logarítmo em `net_bill`",
            "Modelo com transformação por polinômio em `net_bill`",
            "Modelo com logarítmo em `tip_pct`",
            "Modelo com log em ambas as variáveis",
        ]

        # Radio para seleção do modelo a ser visualizado (mostra opções na horizontal)
        selecao_de_modelos = st.radio(
            label="Selecione o modelo que você deseja visualizar",
            options=selecao_de_modelos_options,
            horizontal=True,
        )

        # Estrutura de pares (variável dependente, variável independente) usada para exibição
        estrutura_de_variaveis_utilizadas = [
            ("tip_pct", "net_bill"),
            ("tip_pct", "np.log(net_bill)"),
            ("tip_pct", "np.power(net_bill,2)"),
            ("np.log(tip_pct)", "net_bill"),
            ("np.log(tip_pct)", "np.log(net_bill)"),
        ]

        # Dicionário que mapeia o rótulo do modelo para o par de variáveis que foi usado
        dicionario_visualizacao_var = {
            "Modelo original": estrutura_de_variaveis_utilizadas[0],
            "Modelo com logarítmo em `net_bill`": estrutura_de_variaveis_utilizadas[1],
            "Modelo com transformação por polinômio em `net_bill`": estrutura_de_variaveis_utilizadas[
                2
            ],
            "Modelo com logarítmo em `tip_pct`": estrutura_de_variaveis_utilizadas[3],
            "Modelo com log em ambas as variáveis": estrutura_de_variaveis_utilizadas[
                4
            ],
        }

        # Divide a área em duas colunas: coluna esquerda para gráfico (resíduos) e direita para sumário do modelo
        col1, col2 = st.columns(2)

        # --------------------------------------------------
        # RENDERIZAÇÃO CONDICIONAL: para cada modelo, desenha gráfico de resíduos e exibe summary
        # --------------------------------------------------
        # Para cada opção do radio, o fluxo abaixo cria um scatter plot resíduos x preditor
        # e exibe o summary (texto) do modelo correspondente.
        if selecao_de_modelos == selecao_de_modelos_options[0]:
            # ---------------- Modelo original ----------------
            with col1:
                st.write("#### Distribuição de resíduos do modelo")
                # Mostra quais variáveis foram usadas neste modelo (para contextualizar)
                st.info(
                    f"Variáveis aplicadas na modelagem: {dicionario_visualizacao_var[selecao_de_modelos]}"
                )
                # Cria figura e eixo para o scatterplot dos resíduos
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    x="net_bill",
                    y="resíduos",
                    data=df_modelo_original,
                    alpha=0.75,
                    ax=ax,
                )
                # Linha horizontal em y=0 para referência (resíduo nulo)
                plt.axhline(y=0, color="r", linestyle="--")
                # Salva a figura em um buffer e exibe com st.image
                buff = BytesIO()
                fig.savefig(buff, format="png", bbox_inches="tight")
                buff.seek(0)
                st.image(buff)
                plt.close()
            with col2:
                # Exibe o resumo estatístico do modelo (tabela e coeficientes) como bloco de código
                st.write("#### Sumário do modelo")
                st.code(str(model_regr_tip_pct.summary()), language="text")

        elif selecao_de_modelos == selecao_de_modelos_options[1]:
            # ---------------- Modelo com log(net_bill) ----------------
            with col1:
                st.write("#### Distribuição de resíduos do modelo")
                st.info(
                    f"Variáveis aplicadas na modelagem: {dicionario_visualizacao_var[selecao_de_modelos]}"
                )
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    x="net_bill", y="resíduos", data=df_modelo_01, alpha=0.75, ax=ax
                )
                plt.axhline(y=0, color="r", linestyle="--")
                buff = BytesIO()
                fig.savefig(buff, format="png", bbox_inches="tight")
                buff.seek(0)
                st.image(buff)
                plt.close()
            with col2:
                st.write("#### Sumário do modelo")
                st.code(str(model_01.summary()), language="text")

        elif selecao_de_modelos == selecao_de_modelos_options[2]:
            # ---------------- Modelo polinomial (net_bill^2) ----------------
            with col1:
                st.write("#### Distribuição de resíduos do modelo")
                st.info(
                    f"Variáveis aplicadas na modelagem: {dicionario_visualizacao_var[selecao_de_modelos]}"
                )
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    x="net_bill", y="resíduos", data=df_modelo_02, alpha=0.75, ax=ax
                )
                plt.axhline(y=0, color="r", linestyle="--")
                buff = BytesIO()
                fig.savefig(buff, format="png", bbox_inches="tight")
                buff.seek(0)
                st.image(buff)
                plt.close()
            with col2:
                st.write("#### Sumário do modelo")
                st.code(str(model_02.summary()), language="text")

        elif selecao_de_modelos == selecao_de_modelos_options[3]:
            # ---------------- Modelo com log(tip_pct) ----------------
            with col1:
                st.write("#### Distribuição de resíduos do modelo")
                st.info(
                    f"Variáveis aplicadas na modelagem: {dicionario_visualizacao_var[selecao_de_modelos]}"
                )
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    x="net_bill", y="resíduos", data=df_modelo_03, alpha=0.75, ax=ax
                )
                plt.axhline(y=0, color="r", linestyle="--")
                buff = BytesIO()
                fig.savefig(buff, format="png", bbox_inches="tight")
                buff.seek(0)
                st.image(buff)
                plt.close()
            with col2:
                st.write("#### Sumário do modelo")
                st.code(str(model_03.summary()), language="text")

        elif selecao_de_modelos == selecao_de_modelos_options[4]:
            # ---------------- Modelo com log em ambas as variáveis ----------------
            with col1:
                st.write("#### Distribuição de resíduos do modelo")
                st.info(
                    f"Variáveis aplicadas na modelagem: {dicionario_visualizacao_var[selecao_de_modelos]}"
                )
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    x="net_bill", y="resíduos", data=df_modelo_04, alpha=0.75, ax=ax
                )
                plt.axhline(y=0, color="r", linestyle="--")
                buff = BytesIO()
                fig.savefig(buff, format="png", bbox_inches="tight")
                buff.seek(0)
                st.image(buff)
                plt.close()
            with col2:
                st.write("#### Sumário do modelo")
                st.code(str(model_04.summary()), language="text")

        # --------------------------------------------------
        # COMPARAÇÃO ENTRE MODELOS (MÉTRICAS)
        # --------------------------------------------------
        st.write("---")
        st.subheader("📈 Comparação dos Modelos")

        # Constrói um dicionário com todos os modelos treinados para iteração
        modelos = {
            "Modelo original": model_regr_tip_pct,
            "Modelo com logarítmo em `net_bill`": model_01,
            "Modelo com transformação por polinômio em `net_bill`": model_02,
            "Modelo com logarítmo em `tip_pct`": model_03,
            "Modelo com log em ambas as variáveis": model_04,
        }

        # Lista para armazenar as métricas de cada modelo
        comparacao_modelos = []

        # Para cada modelo, coleta métricas padrão do resultado do statsmodels
        for nome, modelo in modelos.items():
            comparacao_modelos.append(
                {
                    "Modelo": nome,
                    "R²": modelo.rsquared,  # R-squared
                    "R² Ajustado": modelo.rsquared_adj,  # Adjusted R-squared
                    "AIC": modelo.aic,  # Akaike Information Criterion
                    "BIC": modelo.bic,  # Bayesian Information Criterion
                    "F-Statistic": modelo.fvalue,  # Estatística F do modelo
                    "Prob (F-statistic)": modelo.f_pvalue,  # p-valor da estatística F
                    "Número de Observações": modelo.nobs,  # n de observações usadas
                }
            )

        # Converte a lista de dicionários em DataFrame para exibição
        df_comparacao = pd.DataFrame(comparacao_modelos)

        # Exibe a tabela de comparação; aplica formatação numérica e destaca melhores/piores
        st.write("#### Métricas de Comparação")
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
            .highlight_max(
                subset=["R²", "R² Ajustado"], color="lightgreen"
            )  # destaca R² máximos
            .highlight_min(
                subset=["AIC", "BIC"], color="lightgreen"
            ),  # destaca AIC/BIC mínimos
            hide_index=True,
        )

        # --------------------------------------------------
        # DETERMINAR "MELHORES" MODELOS POR DIFERENTES CRITÉRIOS
        # --------------------------------------------------
        # Identifica a linha (modelo) com maior R²
        melhor_r2 = df_comparacao.loc[df_comparacao["R²"].idxmax()]
        # Identifica a linha com maior R² Ajustado
        melhor_r2_ajustado = df_comparacao.loc[df_comparacao["R² Ajustado"].idxmax()]
        # Identifica linha com menor AIC
        menor_aic = df_comparacao.loc[df_comparacao["AIC"].idxmin()]
        # Identifica linha com menor BIC
        menor_bic = df_comparacao.loc[df_comparacao["BIC"].idxmin()]

        # Exibição dos resultados de "melhor por critério" em duas colunas
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

        # --------------------------------------------------
        # LOGICA DE RECOMENDAÇÃO (SIMPLIFICADA)
        # --------------------------------------------------
        st.write("#### 💡 Análise e Recomendação")

        # Cria um conjunto com os modelos que apareceram como "melhor" em cada critério
        modelos_melhores = set(
            [
                melhor_r2["Modelo"],
                melhor_r2_ajustado["Modelo"],
                menor_aic["Modelo"],
                menor_bic["Modelo"],
            ]
        )

        # Se todas as métricas convergirem para o mesmo modelo, recomenda diretamente
        if len(modelos_melhores) == 1:
            modelo_recomendado = list(modelos_melhores)[0]
            st.success(f"**Modelo Recomendado:** {modelo_recomendado}")
            st.info("Todas as métricas convergem para o mesmo modelo como o melhor.")
        else:
            # Caso haja divergência entre critérios, alerta e prioriza R² Ajustado por padrão
            st.warning("**Análise:** As métricas apontam para modelos diferentes.")
            st.info(
                """
            **Interpretação:**
            - **R²**: Explica a variância dos dados (quanto maior, melhor)
            - **R² Ajustado**: R² penalizado pelo número de variáveis
            - **AIC/BIC**: Critérios de informação (quanto menor, melhor)
            - Em caso de divergência, priorize R² Ajustado para comparação justa
            """
            )

            # Recomenda o modelo com maior R² Ajustado como fallback
            modelo_recomendado = melhor_r2_ajustado["Modelo"]
            st.success(
                f"**Modelo Recomendado (baseado no R² Ajustado):** {modelo_recomendado}"
            )

        # --------------------------------------------------
        # EXPLICAÇÃO METODOLÓGICA (EXPANDER)
        # --------------------------------------------------
    with st.expander("🔍 Como foi feita a análise?"):
        # Texto explicativo sobre a metodologia e interpretação das métricas
        st.write(
            """
            **Metodologia de Comparação:**
            
            1. **Coleta de Métricas**: Para cada modelo, coletamos:
               - R² (Coeficiente de Determinação)
               - R² Ajustado (ajustado pelo número de parâmetros)
               - AIC (Critério de Informação de Akaike)
               - BIC (Critério de Informação Bayesiano)
               - F-Statistic e seu p-valor
            
            2. **Identificação dos Melhores**:
               - **Melhor R²**: Maior valor de R²
               - **Melhor R² Ajustado**: Maior valor de R² Ajustado  
               - **Menor AIC**: Menor valor de AIC
               - **Menor BIC**: Menor valor de BIC
            
            3. **Análise de Consenso**:
               - Se todas as métricas apontam para o mesmo modelo → Recomendação clara
               - Se há divergência → Priorizamos o R² Ajustado por ser mais robusto
            
            **Interpretação das Métricas:**
            - **R² > R² Ajustado**: Sempre, pois o R² Ajustado penaliza complexidade
            - **AIC vs BIC**: Ambos penalizam complexidade, BIC é mais rigoroso
            - **F-Statistic**: Testa se o modelo é significativamente melhor que um modelo nulo
            """
        )


# ------------------------------------------------------
# PONTO DE ENTRADA
# ------------------------------------------------------
if __name__ == "__main__":
    main()  # Executa a função principal ao rodar o script
