import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Bibliotecas de aprendizado de máquina
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Bibliotecas de estatística
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

# Configuração da página
st.set_page_config(page_title="Previsão de Renda", layout="wide")

# Título da aplicação
st.title("🔮 Análise de Previsão de Renda")
st.markdown("---")

# Sidebar para configurações
st.sidebar.header("Configurações do Modelo")

# Upload de arquivo
uploaded_file = st.sidebar.file_uploader("📁 Faça upload do arquivo CSV", type=["csv"])


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            # Lê o arquivo
            df = pd.read_csv(uploaded_file)

            # Remove colunas problemáticas
            cols_to_drop = ["Unnamed: 0", "data_ref", "id_cliente"]
            existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
            df.drop(columns=existing_cols_to_drop, inplace=True)

            # Remove linhas com valores missing
            df.dropna(inplace=True)

            return df
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {e}")
            return None
    return None


# Carrega dados se arquivo foi upload
df = load_data(uploaded_file)

if df is not None:
    # Informações básicas do dataset
    st.header("📊 Overview dos Dados")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Informações do Dataset")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Número de variáveis:** {len(df.columns)}")
        st.write(f"**Variável target:** renda")

        # Estatísticas da renda
        st.subheader("Estatísticas da Renda")
        st.write(f"**Média:** R$ {df['renda'].mean():.2f}")
        st.write(f"**Mediana:** R$ {df['renda'].median():.2f}")
        st.write(f"**Desvio Padrão:** R$ {df['renda'].std():.2f}")

    with col2:
        st.subheader("Primeiras linhas")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Tipos de Dados")
        st.dataframe(
            pd.DataFrame(df.dtypes, columns=["Tipo"]), use_container_width=True
        )

    # Divisão dos dados
    x = df.drop(columns=["renda"])
    y = df["renda"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=100
    )

    # Configuração de parâmetros na sidebar
    st.sidebar.subheader("Parâmetros de Regularização")
    alpha_values = st.sidebar.multiselect(
        "Valores de Alpha:",
        [0, 0.001, 0.005, 0.01, 0.05, 0.1],
        default=[0, 0.001, 0.005, 0.01, 0.05, 0.1],
    )

    # Modelo base
    modelo = "renda ~ sexo + posse_de_veiculo + posse_de_imovel + qtd_filhos + tipo_renda + educacao + estado_civil + tipo_residencia + idade + tempo_emprego + qt_pessoas_residencia"

    # Dicionário para armazenar resultados de todos os modelos
    if "model_results" not in st.session_state:
        st.session_state.model_results = {}

    # Abas para diferentes análises
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📈 Ridge Regression",
            "🎯 Lasso Regression",
            "🔍 Stepwise Selection",
            "🌳 Decision Tree",
            "📊 Comparação de Modelos",
        ]
    )

    with tab1:
        st.header("Regularização Ridge (L2)")

        if st.button("Executar Ridge Regression", key="ridge"):
            with st.spinner("Executando Ridge Regression..."):
                dicionario_treinos_ridge = {}
                dicionario_testes_ridge = {}

                # Cálculos para Ridge
                for alpha in alpha_values:
                    # Teste
                    md_teste = smf.ols(modelo, data=x_train.join(y_train))
                    reg = md_teste.fit_regularized(
                        method="elastic_net", refit=False, L1_wt=0, alpha=alpha
                    )
                    y_pred_test = reg.predict(x_test)

                    # Métricas de teste
                    tss_test = ((y_test - y_test.mean()) ** 2).sum()
                    rss_test = ((y_test - y_pred_test) ** 2).sum()
                    r_quadrado_test = 1 - rss_test / tss_test

                    qtd_variaveis_explicativas = len(reg.params) - 1
                    r_quadrado_ajustado_teste = 1 - (1 - r_quadrado_test) * (
                        len(y_test) - 1
                    ) / (len(y_test) - qtd_variaveis_explicativas - 1)

                    residuo_quadrado_test = rss_test / len(y_test)
                    log_vero_test = (
                        -len(y_test)
                        / 2
                        * (np.log(2 * np.pi) + np.log(residuo_quadrado_test) + 1)
                    )
                    aic = 2 * len(reg.params) - 2 * log_vero_test

                    dicionario_testes_ridge[alpha] = {
                        "R2_Teste": round(r_quadrado_test, 3),
                        "R2_Ajustado_Teste": round(r_quadrado_ajustado_teste, 3),
                        "AIC_Teste": round(aic, 2),
                        "MSE_Teste": round(mean_squared_error(y_test, y_pred_test), 2),
                    }

                    # Treino
                    md_treino = smf.ols(modelo, data=x_train.join(y_train))
                    reg = md_treino.fit_regularized(
                        method="elastic_net", refit=False, L1_wt=0, alpha=alpha
                    )
                    y_pred_train = reg.predict(x_train)

                    # Métricas de treino
                    tss_train = ((y_train - y_train.mean()) ** 2).sum()
                    rss_train = ((y_train - y_pred_train) ** 2).sum()
                    r_quadrado_train = 1 - rss_train / tss_train

                    r_quadrado_ajustado_train = 1 - (1 - r_quadrado_train) * (
                        len(y_train) - 1
                    ) / (len(y_train) - qtd_variaveis_explicativas - 1)

                    residuo_quadrado_train = rss_train / len(y_train)
                    log_vero_train = (
                        -len(y_train)
                        / 2
                        * (np.log(2 * np.pi) + np.log(residuo_quadrado_train) + 1)
                    )
                    aic = 2 * len(reg.params) - 2 * log_vero_train

                    dicionario_treinos_ridge[alpha] = {
                        "R2_Treino": round(r_quadrado_train, 3),
                        "R2_Ajustado_Treino": round(r_quadrado_ajustado_train, 3),
                        "AIC_Treino": round(aic, 2),
                        "MSE_Treino": round(
                            mean_squared_error(y_train, y_pred_train), 2
                        ),
                    }

                # Criar DataFrame com resultados
                df_ridge = pd.DataFrame(dicionario_treinos_ridge).T.join(
                    pd.DataFrame(dicionario_testes_ridge).T
                )

                # Encontrar melhor alpha
                best_alpha = df_ridge["R2_Teste"].idxmax()
                best_r2 = df_ridge["R2_Teste"].max()
                best_mse = df_ridge.loc[best_alpha, "MSE_Teste"]
                best_aic = df_ridge.loc[best_alpha, "AIC_Teste"]

                # Armazenar resultados de forma padronizada
                st.session_state.model_results["Ridge"] = {
                    "type": "regularized",
                    "best_alpha": best_alpha,
                    "best_r2": best_r2,
                    "best_mse": best_mse,
                    "best_aic": best_aic,
                    "full_data": df_ridge,
                    "alpha_values": alpha_values,
                }

                st.subheader("Resultados da Ridge Regression")
                st.dataframe(df_ridge, use_container_width=True)

                st.success(f"🎯 Melhor alpha: {best_alpha} (R² = {best_r2:.3f})")

                # Gráfico de comparação
                fig, ax = plt.subplots(1, 2, figsize=(15, 5))

                # R-Quadrado
                ax[0].plot(alpha_values, df_ridge["R2_Treino"], "bo-", label="Treino")
                ax[0].plot(alpha_values, df_ridge["R2_Teste"], "ro-", label="Teste")
                ax[0].set_xlabel("Alpha")
                ax[0].set_ylabel("R-Quadrado")
                ax[0].set_title("R-Quadrado vs Alpha")
                ax[0].legend()
                ax[0].grid(True)

                # AIC
                ax[1].plot(alpha_values, df_ridge["AIC_Teste"], "go-", label="Teste")
                ax[1].set_xlabel("Alpha")
                ax[1].set_ylabel("AIC")
                ax[1].set_title("AIC vs Alpha")
                ax[1].legend()
                ax[1].grid(True)

                st.pyplot(fig)

    with tab2:
        st.header("Regularização Lasso (L1)")

        if st.button("Executar Lasso Regression", key="lasso"):
            with st.spinner("Executando Lasso Regression..."):
                dicionario_treinos_lasso = {}
                dicionario_testes_lasso = {}

                for alpha in alpha_values:
                    # Teste
                    md_teste = smf.ols(modelo, data=x_train.join(y_train))
                    reg = md_teste.fit_regularized(
                        method="elastic_net", refit=True, L1_wt=1, alpha=alpha
                    )
                    y_pred_test = reg.predict(x_test)

                    # Métricas
                    tss_test = ((y_test - y_test.mean()) ** 2).sum()
                    rss_test = ((y_test - y_pred_test) ** 2).sum()
                    r_quadrado_test = 1 - rss_test / tss_test

                    qtd_variaveis_explicativas = len(reg.params) - 1
                    r_quadrado_ajustado_teste = 1 - (1 - r_quadrado_test) * (
                        len(y_test) - 1
                    ) / (len(y_test) - qtd_variaveis_explicativas - 1)

                    residuo_quadrado_test = rss_test / len(y_test)
                    log_vero_test = (
                        -len(y_test)
                        / 2
                        * (np.log(2 * np.pi) + np.log(residuo_quadrado_test) + 1)
                    )
                    aic = 2 * len(reg.params) - 2 * log_vero_test

                    dicionario_testes_lasso[alpha] = {
                        "R2_Teste": round(r_quadrado_test, 3),
                        "R2_Ajustado_Teste": round(r_quadrado_ajustado_teste, 3),
                        "AIC_Teste": round(aic, 2),
                        "MSE_Teste": round(mean_squared_error(y_test, y_pred_test), 2),
                        "Coeficientes_Nao_Zero": len([x for x in reg.params if x != 0]),
                    }

                    # Treino
                    md_treino = smf.ols(modelo, data=x_train.join(y_train))
                    reg = md_treino.fit_regularized(
                        method="elastic_net", refit=True, L1_wt=1, alpha=alpha
                    )
                    y_pred_train = reg.predict(x_train)

                    tss_train = ((y_train - y_train.mean()) ** 2).sum()
                    rss_train = ((y_train - y_pred_train) ** 2).sum()
                    r_quadrado_train = 1 - rss_train / tss_train

                    r_quadrado_ajustado_train = 1 - (1 - r_quadrado_train) * (
                        len(y_train) - 1
                    ) / (len(y_train) - qtd_variaveis_explicativas - 1)

                    residuo_quadrado_train = rss_train / len(y_train)
                    log_vero_train = (
                        -len(y_train)
                        / 2
                        * (np.log(2 * np.pi) + np.log(residuo_quadrado_train) + 1)
                    )
                    aic = 2 * len(reg.params) - 2 * log_vero_train

                    dicionario_treinos_lasso[alpha] = {
                        "R2_Treino": round(r_quadrado_train, 3),
                        "R2_Ajustado_Treino": round(r_quadrado_ajustado_train, 3),
                        "AIC_Treino": round(aic, 2),
                        "MSE_Treino": round(
                            mean_squared_error(y_train, y_pred_train), 2
                        ),
                    }

                df_lasso = pd.DataFrame(dicionario_treinos_lasso).T.join(
                    pd.DataFrame(dicionario_testes_lasso).T
                )

                # Encontrar melhor alpha
                best_alpha = df_lasso["R2_Teste"].idxmax()
                best_r2 = df_lasso["R2_Teste"].max()
                best_mse = df_lasso.loc[best_alpha, "MSE_Teste"]
                best_aic = df_lasso.loc[best_alpha, "AIC_Teste"]
                coef_nao_zero = df_lasso.loc[best_alpha, "Coeficientes_Nao_Zero"]

                # Armazenar resultados de forma padronizada
                st.session_state.model_results["Lasso"] = {
                    "type": "regularized",
                    "best_alpha": best_alpha,
                    "best_r2": best_r2,
                    "best_mse": best_mse,
                    "best_aic": best_aic,
                    "coeficientes_nao_zero": coef_nao_zero,
                    "full_data": df_lasso,
                    "alpha_values": alpha_values,
                }

                st.subheader("Resultados da Lasso Regression")
                st.dataframe(df_lasso, use_container_width=True)

                st.success(
                    f"🎯 Melhor alpha: {best_alpha} (R² = {best_r2:.3f}, Coef. não-zero: {coef_nao_zero})"
                )

                # Gráfico de coeficientes não-zero
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(
                    alpha_values,
                    df_lasso["Coeficientes_Nao_Zero"],
                    "go-",
                    linewidth=2,
                    markersize=8,
                )
                ax.set_xlabel("Alpha")
                ax.set_ylabel("Número de Coeficientes Não-Zero")
                ax.set_title("Seleção de Variáveis - Lasso")
                ax.grid(True)
                st.pyplot(fig)

    with tab3:
        st.header("Seleção Stepwise")

        if st.button("Executar Seleção Stepwise", key="stepwise"):
            with st.spinner(
                "Executando seleção stepwise... (pode demorar alguns minutos)"
            ):
                try:
                    # Preparar dados para stepwise - garantir tipos numéricos
                    X_train_dummies = pd.get_dummies(x_train, drop_first=True)
                    X_test_dummies = pd.get_dummies(x_test, drop_first=True)

                    X_train_dummies.columns = X_train_dummies.columns.str.replace(
                        " ", "_", regex=True
                    )
                    X_test_dummies.columns = X_test_dummies.columns.str.replace(
                        " ", "_", regex=True
                    )

                    # Converter para float para evitar problemas com statsmodels
                    X_train_dummies = X_train_dummies.astype(float)
                    X_test_dummies = X_test_dummies.astype(float)
                    y_train_clean = y_train.astype(float)

                    # Função stepwise corrigida
                    def stepwise_selection_corrigida(
                        X, y, initial_list=[], threshold_in=0.05, threshold_out=0.05
                    ):
                        included = list(initial_list)
                        max_iter = 50  # Prevenir loop infinito
                        iter_count = 0

                        while iter_count < max_iter:
                            iter_count += 1
                            changed = False

                            # Forward step
                            excluded = list(set(X.columns) - set(included))
                            if not excluded:
                                break

                            new_pval = pd.Series(index=excluded, dtype=float)
                            for new_column in excluded:
                                try:
                                    # Garantir que os dados são numéricos
                                    X_temp = X[included + [new_column]].copy()
                                    X_temp = sm.add_constant(X_temp.astype(float))
                                    y_temp = y.astype(float)

                                    model = sm.OLS(y_temp, X_temp).fit()
                                    new_pval[new_column] = model.pvalues[new_column]
                                except Exception as e:
                                    new_pval[new_column] = 1.0  # Valor alto se der erro

                            best_pval = new_pval.min()
                            if best_pval < threshold_in:
                                best_feature = new_pval.idxmin()
                                included.append(best_feature)
                                changed = True

                            # Backward step
                            if included:
                                try:
                                    X_temp = X[included].copy()
                                    X_temp = sm.add_constant(X_temp.astype(float))
                                    y_temp = y.astype(float)

                                    model = sm.OLS(y_temp, X_temp).fit()
                                    pvalues = model.pvalues.iloc[
                                        1:
                                    ]  # Excluir intercept

                                    if not pvalues.empty:
                                        worst_pval = pvalues.max()
                                        if worst_pval > threshold_out:
                                            worst_feature = pvalues.idxmax()
                                            included.remove(worst_feature)
                                            changed = True
                                except Exception as e:
                                    st.warning(f"Erro no backward step: {e}")

                            if not changed:
                                break

                        return included

                    variaveis_selecionadas = stepwise_selection_corrigida(
                        X_train_dummies, y_train_clean
                    )

                    st.subheader("Variáveis Selecionadas")
                    if variaveis_selecionadas:
                        st.write(
                            f"**Número de variáveis selecionadas:** {len(variaveis_selecionadas)}"
                        )
                        st.write(variaveis_selecionadas)

                        # Testar com as variáveis selecionadas
                        X_train_selected = X_train_dummies[variaveis_selecionadas]
                        X_test_selected = X_test_dummies[variaveis_selecionadas]

                        string_variaveis = " + ".join(variaveis_selecionadas)

                        dicionario_indicadores = {}

                        for alpha in alpha_values:
                            try:
                                md_treino = smf.ols(
                                    f"renda ~ {string_variaveis}",
                                    data=X_train_selected.join(y_train),
                                )
                                reg = md_treino.fit_regularized(
                                    method="elastic_net",
                                    refit=True,
                                    L1_wt=1,
                                    alpha=alpha,
                                )

                                y_pred_test = reg.predict(X_test_selected)

                                r_quadrado = r2_score(y_test, y_pred_test)
                                r_quadrado_ajustado = 1 - (
                                    (1 - r_quadrado) * (len(y_test) - 1)
                                ) / (len(y_test) - len(X_test_selected.columns) - 1)

                                rss = np.power(y_test - y_pred_test, 2).sum()
                                log_vero_test = (
                                    -len(y_test)
                                    / 2
                                    * (
                                        np.log(2 * np.pi)
                                        + np.log(rss / len(y_test))
                                        + 1
                                    )
                                )

                                aic = 2 * len(reg.params) - 2 * log_vero_test

                                dicionario_indicadores[alpha] = {
                                    "R2_Teste": round(r_quadrado, 3),
                                    "R2_Ajustado_Teste": round(r_quadrado_ajustado, 3),
                                    "AIC_Teste": round(aic, 2),
                                    "MSE_Teste": round(
                                        mean_squared_error(y_test, y_pred_test), 2
                                    ),
                                }
                            except Exception as e:
                                st.warning(f"Erro com alpha {alpha}: {e}")
                                continue

                        if dicionario_indicadores:
                            df_stepwise = pd.DataFrame.from_dict(
                                dicionario_indicadores, orient="index"
                            )

                            # Encontrar melhor alpha
                            best_alpha = df_stepwise["R2_Teste"].idxmax()
                            best_r2 = df_stepwise.loc[best_alpha, "R2_Teste"]
                            best_mse = df_stepwise.loc[best_alpha, "MSE_Teste"]
                            best_aic = df_stepwise.loc[best_alpha, "AIC_Teste"]

                            # Armazenar resultados de forma padronizada
                            st.session_state.model_results["Stepwise"] = {
                                "type": "stepwise",
                                "best_alpha": best_alpha,
                                "best_r2": best_r2,
                                "best_mse": best_mse,
                                "best_aic": best_aic,
                                "selected_features": variaveis_selecionadas,
                                "full_data": df_stepwise,
                                "alpha_values": [
                                    k for k in dicionario_indicadores.keys()
                                ],
                            }

                            st.subheader("Resultados com Variáveis Selecionadas")
                            st.dataframe(df_stepwise, use_container_width=True)
                            st.success(
                                f"🎯 Melhor alpha: {best_alpha} (R² = {best_r2:.3f})"
                            )
                        else:
                            st.error(
                                "Não foi possível calcular métricas para nenhum alpha."
                            )
                    else:
                        st.warning(
                            "Nenhuma variável foi selecionada no processo stepwise."
                        )

                except Exception as e:
                    st.error(f"Erro no processo stepwise: {e}")

    with tab4:
        st.header("Árvore de Decisão")

        if st.button("Executar Decision Tree", key="tree"):
            with st.spinner("Treinando Decision Tree..."):
                # Preparar dados
                X_train_dummies = pd.get_dummies(x_train, drop_first=True)
                X_test_dummies = pd.get_dummies(x_test, drop_first=True)

                X_train_dummies.columns = X_train_dummies.columns.str.replace(
                    " ", "_", regex=True
                )
                X_test_dummies.columns = X_test_dummies.columns.str.replace(
                    " ", "_", regex=True
                )

                # Treinar modelo
                regr = DecisionTreeRegressor(random_state=42)
                regr.fit(X_train_dummies, y_train)

                # Previsões
                y_pred_train = regr.predict(X_train_dummies)
                y_pred_test = regr.predict(X_test_dummies)

                # Métricas
                r2_train = r2_score(y_train, y_pred_train)
                r2_test = r2_score(y_test, y_pred_test)
                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_test = mean_squared_error(y_test, y_pred_test)

                # Calcular AIC aproximado
                n = len(y_test)
                rss = np.sum((y_test - y_pred_test) ** 2)
                k = X_train_dummies.shape[1] + 1  # número de parâmetros
                aic = n * np.log(rss / n) + 2 * k

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("R² Treino", f"{r2_train:.3f}")
                    st.metric("MSE Treino", f"{mse_train:.2f}")

                with col2:
                    st.metric("R² Teste", f"{r2_test:.3f}")
                    st.metric("MSE Teste", f"{mse_test:.2f}")
                    st.metric("AIC Teste", f"{aic:.2f}")

                # Armazenar resultados de forma padronizada
                st.session_state.model_results["Decision_Tree"] = {
                    "type": "tree",
                    "best_r2": r2_test,
                    "best_mse": mse_test,
                    "best_aic": aic,
                    "feature_importance": pd.DataFrame(
                        {
                            "feature": X_train_dummies.columns,
                            "importance": regr.feature_importances_,
                        }
                    ).sort_values("importance", ascending=False),
                }

                st.success(f"🎯 Decision Tree (R² = {r2_test:.3f})")

                # Importância das variáveis
                feature_importance = st.session_state.model_results["Decision_Tree"][
                    "feature_importance"
                ]

                st.subheader("Importância das Variáveis")

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(
                    data=feature_importance.head(10), x="importance", y="feature", ax=ax
                )
                ax.set_title("Top 10 Variáveis Mais Importantes")
                st.pyplot(fig)

    with tab5:
        st.header("📊 Comparação de Modelos")

        if st.button("Comparar Todos os Modelos", key="compare"):
            if not st.session_state.model_results:
                st.warning("Execute pelo menos um modelo primeiro para comparar.")
            else:
                st.subheader("Performance dos Modelos")

                comparison_data = []

                for model_name, results in st.session_state.model_results.items():
                    # Extrair métricas de forma segura
                    try:
                        # Para modelos com alpha (Ridge, Lasso, Stepwise)
                        if "best_alpha" in results:
                            best_r2 = results["best_r2"]
                            best_mse = results["best_mse"]
                            best_aic = results["best_aic"]
                            alpha_info = f"α={results['best_alpha']}"
                        # Para Decision Tree
                        else:
                            best_r2 = results.get("r2_teste", results.get("best_r2", 0))
                            best_mse = results.get(
                                "mse_teste", results.get("best_mse", 0)
                            )
                            best_aic = results.get(
                                "aic_teste", results.get("best_aic", 0)
                            )
                            alpha_info = "N/A"

                        comparison_data.append(
                            {
                                "Modelo": model_name.replace("_", " "),
                                "Alpha": alpha_info,
                                "R² Teste": round(best_r2, 4),
                                "MSE Teste": round(best_mse, 2),
                                "AIC Teste": round(best_aic, 2),
                            }
                        )
                    except Exception as e:
                        st.warning(f"Erro ao processar modelo {model_name}: {e}")
                        continue

                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)

                    # Ordenar por R² Teste (maior primeiro)
                    df_comparison = df_comparison.sort_values(
                        "R² Teste", ascending=False
                    )

                    st.dataframe(df_comparison, use_container_width=True)

                    # Gráfico de comparação
                    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                    # Gráfico de R²
                    bars1 = axes[0].bar(
                        df_comparison["Modelo"],
                        df_comparison["R² Teste"],
                        color=["blue", "green", "red", "orange", "purple"],
                    )
                    axes[0].set_title("Comparação de R² entre Modelos")
                    axes[0].set_ylabel("R² Score")
                    axes[0].tick_params(axis="x", rotation=45)

                    # Adicionar valores nas barras
                    for bar, v in zip(bars1, df_comparison["R² Teste"]):
                        axes[0].text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.3f}",
                            ha="center",
                            va="bottom",
                        )

                    # Gráfico de MSE
                    bars2 = axes[1].bar(
                        df_comparison["Modelo"],
                        df_comparison["MSE Teste"],
                        color=["blue", "green", "red", "orange", "purple"],
                    )
                    axes[1].set_title("Comparação de MSE entre Modelos")
                    axes[1].set_ylabel("Mean Squared Error")
                    axes[1].tick_params(axis="x", rotation=45)

                    # Adicionar valores nas barras
                    for bar, v in zip(bars2, df_comparison["MSE Teste"]):
                        axes[1].text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.0f}",
                            ha="center",
                            va="bottom",
                        )

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Recomendação do melhor modelo
                    best_model_row = df_comparison.iloc[0]
                    st.success(
                        f"🎯 **Melhor modelo:** {best_model_row['Modelo']} (R² = {best_model_row['R² Teste']:.3f})"
                    )

                    # Mostrar informações adicionais
                    if (
                        "Stepwise" in st.session_state.model_results
                        and "selected_features"
                        in st.session_state.model_results["Stepwise"]
                    ):
                        st.subheader("Variáveis Selecionadas no Stepwise")
                        st.write(
                            st.session_state.model_results["Stepwise"][
                                "selected_features"
                            ]
                        )

                    if (
                        "Lasso" in st.session_state.model_results
                        and "coeficientes_nao_zero"
                        in st.session_state.model_results["Lasso"]
                    ):
                        st.subheader("Coeficientes Não-Zero no Lasso")
                        st.write(
                            f"Número de coeficientes não-zero: {st.session_state.model_results['Lasso']['coeficientes_nao_zero']}"
                        )
                else:
                    st.warning("Nenhum dado disponível para comparação.")

else:
    st.info(
        "👆 Por favor, faça upload do arquivo CSV no menu lateral para começar a análise."
    )

# Informações finais
st.sidebar.markdown("---")
st.sidebar.info(
    """
**Instruções:**
1. Faça upload do arquivo CSV no menu lateral
2. Selecione os valores de alpha desejados
3. Navegue pelas abas para executar diferentes análises
4. Use a aba 'Comparação' para ver o desempenho de todos os modelos

**Alpha Values:** 0, 0.001, 0.005, 0.01, 0.05, 0.1
"""
)
