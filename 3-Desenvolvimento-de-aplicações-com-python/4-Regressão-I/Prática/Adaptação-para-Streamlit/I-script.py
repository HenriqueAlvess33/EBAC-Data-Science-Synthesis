import pandas as pd
import seaborn as sns
import streamlit as st
from seaborn import load_dataset
from io import BytesIO

import matplotlib.pyplot as plt

import numpy as np

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
    # Layout do cabeçalho: três colunas com proporções customizadas
    titulo_col_1, titulo_col_2, titulo_col_3 = st.columns([1.5, 4, 1])
    # Título centralizado (na coluna do meio)
    titulo_col_2.title("Modelo de regressão linear baseado no dataset `tips`")

    tips = sns.load_dataset("tips")
    tips["tip_pct"] = tips["tip"] / (tips["total_bill"] - tips["tip"])
    tips["net_bill"] = tips["total_bill"] - tips["tip"]

    with st.expander(
        "📊 Explicação sobre objetivo da atividade e conceito da base de dados",
        expanded=True,
    ):
        st.write(
            """
        ## 🎯 Objetivo da Atividade
        
        Esta aplicação tem como objetivo demonstrar a criação e comparação de **modelos de regressão linear** 
        para prever o comportamento de gorjetas em restaurantes, utilizando o famoso dataset `tips` do Seaborn.
        
        ### 📈 Conceitos Abordados:
        
        **1. Regressão Linear Simples**
        - Modelagem da relação entre uma variável independente (net_bill) e uma variável dependente (tip ou tip_pct)
        - Equação: `y = β₀ + β₁*x + ε`
        
        **2. Métricas de Avaliação**
        - **R-Quadrado**: Mede a proporção da variância na variável dependente que é previsível a partir da variável independente
        - **Mean Squared Error (MSE)**: Mede a qualidade do ajuste do modelo (quanto menor, melhor)
        
        **3. Comparação de Modelos**
        - Análise de qual abordagem (valor absoluto vs. percentual) produz melhores previsões
        - Uso de deltas para comparação direta entre modelos
        """
        )

        st.write("### 💰 Sobre a Base de Dados `tips`")

        col_info1, col_info2 = st.columns(2)

        with col_info1:
            st.write(
                """
            **Variáveis Originais:**
            - `total_bill`: Valor total da conta (USD)
            - `tip`: Valor da gorjeta (USD)
            - `sex`: Gênero do cliente
            - `smoker`: Cliente fumante?
            - `day`: Dia da semana
            - `time`: Período (Almoço/Jantar)
            - `size`: Número de pessoas na mesa
            """
            )

        with col_info2:
            st.write(
                """
            **Variáveis Criadas:**
            - `net_bill`: total_bill - tip (conta líquida)
            - `tip_pct`: tip / net_bill (percentual da gorjeta)
            
            **Linha de Referência:**
            - **Linha vermelha**: Representa 10% de gorjeta (benchmark comum)
            """
            )

        st.write(
            """
        ### 🔍 Contexto do Problema
        
        Restaurantes e garçons frequentemente querem entender:
        - Como o valor da conta influencia o valor da gorjeta?
        - É melhor modelar gorjetas como valor absoluto ou percentual?
        - Qual modelo tem melhor poder preditivo?
        
        Esta análise ajuda a responder essas questões através de modelagem estatística.
        """
        )

    with st.expander("Criação de performance do modelo de regressão", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            selecao_variavel_interesse = st.radio(
                options=["tip", "tip_pct"],
                label="Selecione a variável de interesse",
                help="Selecione a variável que você deseja ver em um gráfico de distribuição",
                horizontal=True,
            )

            model_regr_tip = smf.ols(f"tip~net_bill", data=tips).fit()

            model_regr_tip_pct = smf.ols(f"tip_pct~net_bill", data=tips).fit()

            sns.regplot(
                x="net_bill",
                y=selecao_variavel_interesse,
                data=tips.loc[tips["tip_pct"] < 0.5],
            )

            x = tips["net_bill"]
            y = [0.1] * len(x)

            plt.plot(x, y, "-r", label="linha dos 10%")

            plt.xlim(left=0)
            plt.ylim(bottom=0)

            buff = BytesIO()
            plt.savefig(buff, format="png", bbox_inches="tight")
            buff.seek(0)
            st.write(
                f"#### Gráfico de dispersão com `{selecao_variavel_interesse}` no eixo Y"
            )
            st.image(buff)
            with col2:
                st.write("#### Dados de performance dos dois modelos")
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    st.write("##### Regressão com `tip`como variável de interesse")
                    st.metric(
                        "R-Quadrado",
                        round(model_regr_tip.rsquared, 3),
                    )
                    st.metric(
                        "Mean Square Error",
                        round(model_regr_tip.mse_model, 3),
                    )
                with sub_col2:
                    st.write("##### Regressão com `tip_pct`como variável de interesse")

                    # Cálculo dos deltas em relação ao modelo da primeira coluna
                    r_squared_delta = round(
                        model_regr_tip_pct.rsquared - model_regr_tip.rsquared, 3
                    )
                    mse_delta = round(
                        model_regr_tip_pct.mse_model - model_regr_tip.mse_model, 3
                    )

                    st.metric(
                        "R-Quadrado",
                        round(model_regr_tip_pct.rsquared, 3),
                        delta=r_squared_delta,
                    )
                    st.metric(
                        "Mean Squared Error",
                        round(model_regr_tip_pct.mse_model, 3),
                        delta=mse_delta,
                    )

    # ------------------------------------------------------
    # EXPANDER COM PREDICT INTERATIVO
    # ------------------------------------------------------
    with st.expander("🔮 Predict Interativo", expanded=True):
        st.write("### Faça uma previsão com os modelos treinados")

        # Criando colunas para organização
        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            st.write("#### Configuração da Previsão")

            # Seleção do modelo
            modelo_selecionado = st.radio(
                "Selecione o modelo para previsão:",
                options=["Modelo Tip", "Modelo Tip Percentage"],
                index=0,
                help="Escolha qual modelo usar para a previsão",
            )

            # Input do valor net_bill
            net_bill_input = st.slider(
                "Valor da conta líquida (net_bill):",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=0.5,
                help="Selecione o valor da conta líquida para fazer a previsão",
            )

            # Botão para fazer a previsão
            fazer_predicao = st.button("Fazer Previsão", type="primary")

        with pred_col2:
            st.write("#### Resultado da Previsão")

            if fazer_predicao:
                # Criar DataFrame para a previsão
                dados_predicao = pd.DataFrame({"net_bill": [net_bill_input]})

                if modelo_selecionado == "Modelo Tip":
                    # Fazer previsão com modelo tip
                    predicao = model_regr_tip.predict(dados_predicao)
                    resultado = predicao.iloc[0]

                    st.success(f"**Previsão de Gorjeta (tip):** ${resultado:.2f}")

                    # Mostrar equação do modelo
                    intercept = model_regr_tip.params["Intercept"]
                    coef = model_regr_tip.params["net_bill"]
                    st.write(
                        f"**Equação do modelo:** `tip = {intercept:.3f} + {coef:.3f} * net_bill`"
                    )

                    # Cálculo detalhado
                    st.write("**Cálculo detalhado:**")
                    st.write(
                        f"`{intercept:.3f} + {coef:.3f} × {net_bill_input} = {resultado:.3f}`"
                    )

                else:
                    # Fazer previsão com modelo tip_pct
                    predicao = model_regr_tip_pct.predict(dados_predicao)
                    resultado = predicao.iloc[0]

                    st.success(
                        f"**Previsão de Percentual de Gorjeta (tip_pct):** {resultado:.3f}"
                    )

                    # Mostrar equação do modelo
                    intercept = model_regr_tip_pct.params["Intercept"]
                    coef = model_regr_tip_pct.params["net_bill"]
                    st.write(
                        f"**Equação do modelo:** `tip_pct = {intercept:.3f} + {coef:.3f} * net_bill`"
                    )

                    # Cálculo detalhado
                    st.write("**Cálculo detalhado:**")
                    st.write(
                        f"`{intercept:.3f} + {coef:.3f} × {net_bill_input} = {resultado:.3f}`"
                    )

                # Informações adicionais
                st.info(
                    """
                **Interpretação:**
                - **Modelo Tip**: Previsão do valor absoluto da gorjeta em dólares
                - **Modelo Tip Percentage**: Previsão do percentual da gorjeta em relação ao valor líquido da conta
                """
                )
            else:
                st.info(
                    "Configure os parâmetros ao lado e clique em 'Fazer Previsão' para ver os resultados."
                )


# ------------------------------------------------------
# PONTO DE ENTRADA
# ------------------------------------------------------
if __name__ == "__main__":
    main()  # Executa a função principal ao rodar o script
