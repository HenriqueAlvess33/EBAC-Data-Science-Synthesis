import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuração da página
st.set_page_config(page_title="Análise de Previsão de Renda", layout="wide")

st.title("📊 Análise de Previsão de Renda")
st.markdown("---")


# Carregar dados
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("./input/previsao_de_renda.csv")
        return df
    except FileNotFoundError:
        st.error("Arquivo 'previsao_de_renda.csv' não encontrado na pasta 'input'.")
        return None


df = load_data()

if df is not None:
    # Sidebar para navegação
    st.sidebar.title("Navegação")
    sections = [
        "Visão Geral dos Dados",
        "Análise de Frequência das Variáveis",
        "Modelo de Regressão Completo",
        "Análise de Significância",
        "Modelo Final Otimizado",
    ]
    selected_section = st.sidebar.radio("Selecione a seção:", sections)

    # Seção 1: Visão Geral dos Dados
    if selected_section == "Visão Geral dos Dados":
        st.header("📋 Visão Geral dos Dados")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Informações do Dataset")
            buffer = st.container()
            with buffer:
                st.text("Estrutura do DataFrame:")
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Número de colunas:** {len(df.columns)}")

        with col2:
            st.subheader("Primeiras Linhas")
            st.dataframe(df.head(), use_container_width=True)

        st.subheader("Estatísticas Descritivas")
        st.dataframe(df.describe(), use_container_width=True)

    # Seção 2: Análise de Frequência
    elif selected_section == "Análise de Frequência das Variáveis":
        st.header("📈 Análise de Frequência das Variáveis")

        lista_de_variaveis = df.columns.to_list()[3:]
        dicionario_de_variaveis = {}
        dicionario_de_variaveis_max = {}

        for variavel in lista_de_variaveis:
            if variavel in ["renda", "idade", "tempo_emprego"]:
                continue
            contagem = df[variavel].value_counts()
            dicionario_de_variaveis[variavel] = pd.DataFrame(
                {"index": contagem.index, "valor": contagem.values}
            )
            dicionario_de_variaveis_max[variavel] = dicionario_de_variaveis[variavel][
                dicionario_de_variaveis[variavel]["valor"]
                == dicionario_de_variaveis[variavel]["valor"].max()
            ]

        dataframe_de_contagem = pd.concat(dicionario_de_variaveis, axis=1)
        dataframe_de_contagem_max = pd.concat(dicionario_de_variaveis_max, axis=1)

        st.subheader("Categorias Mais Frequentes por Variável")
        st.dataframe(dataframe_de_contagem_max, use_container_width=True)

        # Gráficos de frequência
        st.subheader("Distribuição das Variáveis Categóricas")

        variaveis_categoricas = [
            var
            for var in lista_de_variaveis
            if var not in ["renda", "idade", "tempo_emprego"]
        ]

        selected_var = st.selectbox(
            "Selecione uma variável para visualizar:", variaveis_categoricas
        )

        if selected_var:
            fig, ax = plt.subplots(figsize=(10, 6))
            df[selected_var].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Distribuição de {selected_var}")
            ax.set_xlabel(selected_var)
            ax.set_ylabel("Frequência")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Seção 3: Modelo de Regressão Completo
    elif selected_section == "Modelo de Regressão Completo":
        st.header("🔮 Modelo de Regressão Completo")

        st.subheader("Fórmula do Modelo")
        formula_completa = """np.log(renda) ~ C(sexo, Treatment("F")) + 
        C(posse_de_veiculo, Treatment(False)) + 
        C(posse_de_imovel, Treatment(True)) + 
        C(qtd_filhos, Treatment(0)) + 
        C(tipo_renda, Treatment("Assalariado")) + 
        C(educacao, Treatment("Secundário")) + 
        C(estado_civil, Treatment("Casado")) + 
        C(tipo_residencia, Treatment("Casa")) + 
        idade + tempo_emprego"""

        st.code(formula_completa, language="python")

        if st.button("Executar Modelo Completo"):
            with st.spinner("Ajustando modelo..."):
                y, x = patsy.dmatrices(formula_completa, df)
                reg = sm.OLS(y, x).fit()

                st.subheader("Resumo do Modelo")

                # Métricas principais
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R²", f"{reg.rsquared:.4f}")
                with col2:
                    st.metric("R² Ajustado", f"{reg.rsquared_adj:.4f}")
                with col3:
                    st.metric("Observações", reg.nobs)

                # Tabela de coeficientes
                st.subheader("Coeficientes do Modelo")
                coef_df = pd.DataFrame(
                    {
                        "Coeficiente": reg.params,
                        "Erro Padrão": reg.bse,
                        "t-value": reg.tvalues,
                        "p-value": reg.pvalues,
                    }
                )
                st.dataframe(coef_df.style.format("{:.4f}"), use_container_width=True)

    # Seção 4: Análise de Significância
    elif selected_section == "Análise de Significância":
        st.header("📊 Análise de Significância Estatística")

        # Executar modelo para análise de p-values
        formula_analise = """np.log(renda) ~ C(sexo, Treatment("F")) + 
        C(posse_de_veiculo, Treatment(False)) + 
        C(posse_de_imovel, Treatment(True)) + 
        C(qtd_filhos, Treatment(0)) + 
        C(tipo_renda, Treatment("Assalariado")) + 
        C(educacao, Treatment("Secundário")) + 
        C(estado_civil, Treatment("Casado")) + 
        C(tipo_residencia, Treatment("Casa")) + 
        idade + tempo_emprego"""

        y, x = patsy.dmatrices(formula_analise, df)
        reg = sm.OLS(y, x).fit()

        p_values = pd.Series(reg.pvalues, index=x.design_info.column_names)
        variaveis_significativas = p_values[p_values <= 0.05].index.tolist()
        variaveis_nao_significativas = p_values[p_values > 0.05].index.tolist()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Variáveis Significativas (p ≤ 0.05)")
            for var in variaveis_significativas:
                st.write(f"• {var} (p = {p_values[var]:.4f})")

        with col2:
            st.subheader("❌ Variáveis Não Significativas (p > 0.05)")
            for var in variaveis_nao_significativas:
                st.write(f"• {var} (p = {p_values[var]:.4f})")

        # Gráfico de p-values
        st.subheader("Distribuição dos P-values")
        fig, ax = plt.subplots(figsize=(10, 6))
        significant_data = p_values[p_values <= 0.05]
        non_significant_data = p_values[p_values > 0.05]

        ax.scatter(
            significant_data.index,
            significant_data.values,
            color="green",
            label="Significativo (p ≤ 0.05)",
            s=100,
        )
        ax.scatter(
            non_significant_data.index,
            non_significant_data.values,
            color="red",
            label="Não Significativo (p > 0.05)",
            s=100,
        )

        ax.axhline(
            y=0.05,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Limite de significância (0.05)",
        )
        ax.set_ylabel("p-value")
        ax.set_xlabel("Variáveis")
        ax.set_xticklabels(p_values.index, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Seção 5: Modelo Final Otimizado
    elif selected_section == "Modelo Final Otimizado":
        st.header("🎯 Modelo Final Otimizado")

        st.subheader("Fórmula do Modelo Otimizado")
        formula_otimizada = """np.log(renda) ~ C(sexo, Treatment("F")) + 
        C(posse_de_veiculo, Treatment(False)) + 
        C(posse_de_imovel, Treatment(True)) + 
        idade + tempo_emprego"""

        st.code(formula_otimizada, language="python")

        if st.button("Executar Modelo Otimizado"):
            with st.spinner("Ajustando modelo otimizado..."):
                y, x = patsy.dmatrices(formula_otimizada, df)
                reg = sm.OLS(y, x).fit()

                st.subheader("Resumo do Modelo Otimizado")

                # Comparação de métricas
                st.subheader("Comparação de Desempenho")

                # Modelo completo para comparação
                formula_completa = """np.log(renda) ~ C(sexo, Treatment("F")) + 
                C(posse_de_veiculo, Treatment(False)) + 
                C(posse_de_imovel, Treatment(True)) + 
                C(qtd_filhos, Treatment(0)) + 
                C(tipo_renda, Treatment("Assalariado")) + 
                C(educacao, Treatment("Secundário")) + 
                C(estado_civil, Treatment("Casado")) + 
                C(tipo_residencia, Treatment("Casa")) + 
                idade + tempo_emprego"""

                y_full, x_full = patsy.dmatrices(formula_completa, df)
                reg_full = sm.OLS(y_full, x_full).fit()

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "R²",
                        f"{reg.rsquared:.4f}",
                        f"{(reg.rsquared - reg_full.rsquared):.4f}",
                    )
                with col2:
                    st.metric(
                        "R² Ajustado",
                        f"{reg.rsquared_adj:.4f}",
                        f"{(reg.rsquared_adj - reg_full.rsquared_adj):.4f}",
                    )
                with col3:
                    st.metric(
                        "Número de Variáveis",
                        len(reg.params),
                        f"-{len(reg_full.params) - len(reg.params)}",
                    )

                # Coeficientes do modelo otimizado
                st.subheader("Coeficientes do Modelo Otimizado")
                coef_df = pd.DataFrame(
                    {
                        "Coeficiente": reg.params,
                        "Erro Padrão": reg.bse,
                        "t-value": reg.tvalues,
                        "p-value": reg.pvalues,
                    }
                )
                st.dataframe(coef_df.style.format("{:.4f}"), use_container_width=True)

                # Resíduos
                st.subheader("Análise de Resíduos")
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                # Plot 1: Resíduos vs Valores Ajustados
                axes[0].scatter(reg.fittedvalues, reg.resid, alpha=0.6)
                axes[0].axhline(y=0, color="red", linestyle="--")
                axes[0].set_xlabel("Valores Ajustados")
                axes[0].set_ylabel("Resíduos")
                axes[0].set_title("Resíduos vs Valores Ajustados")

                # Plot 2: QQ Plot
                sm.qqplot(reg.resid, line="45", ax=axes[1])
                axes[1].set_title("Q-Q Plot dos Resíduos")

                plt.tight_layout()
                st.pyplot(fig)

    # Footer
    st.markdown("---")
    st.markdown("**Desenvolvido com Streamlit** | Análise de Previsão de Renda")

else:
    st.error(
        "Não foi possível carregar os dados. Verifique se o arquivo 'previsao_de_renda.csv' está na pasta 'input'."
    )
