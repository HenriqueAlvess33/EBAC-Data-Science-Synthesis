import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


# ------------------------------------------------------
# FUNÇÃO PARA CARREGAR DADOS
# ------------------------------------------------------
def load_data(conjunto_de_dados):
    """
    Carrega um arquivo CSV enviado pelo usuário.
    - Retorna um DataFrame pandas.
    - Caso haja erro na leitura, exibe uma mensagem na interface.
    Obs.: o decorador @st.cache_data (mencionado na docstring) não foi aplicado,
    mas poderia ser usado para evitar recarregar o arquivo a cada interação.
    """
    try:
        # Lê o arquivo CSV em um DataFrame e retorna
        # -> Observação: aqui é chamado pd.read_csv, então se o arquivo for .xlsx
        #    esta chamada vai falhar; no seu projeto original isso não é problema.
        return pd.read_csv(conjunto_de_dados)  # Lê o arquivo CSV em um DataFrame
    except Exception as e:
        # Se ocorrer qualquer exceção durante a leitura, exibe a mensagem de erro no app
        st.error(f"Não foi possível carregar o arquivo selecionado. {e}")


# ------------------------------------------------------
# CONFIGURAÇÃO INICIAL DA PÁGINA STREAMLIT
# ------------------------------------------------------
st.set_page_config(
    page_title="Preparando a base",  # Título exibido na aba do navegador
    layout="wide",  # Usa toda a largura da tela
    initial_sidebar_state="expanded",  # Barra lateral aberta por padrão
    page_icon="varig_icon.png",  # Ícone exibido na aba
)


# ------------------------------------------------------
# FUNÇÃO PRINCIPAL DO APP
# ------------------------------------------------------
def main():
    # Upload do arquivo pelo usuário, exibido na barra lateral
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Faça o upload do dataframe previsao_de_renda.csv", type=["csv", "xlsx"]
        )
        # file_uploader cria um widget que aceita arquivos CSV e XLSX (aqui aceita ambos)

    # Se um arquivo for carregado:
    if uploaded_file is not None:
        # Chama a função de carregamento para transformar o conteúdo em DataFrame
        previsao_de_renda = load_data(uploaded_file)  # Lê os dados

        # ---- Remoção de colunas indesejadas ----
        colunas_para_remover = ["Unnamed: 0", "mau", "index"]
        # Cria lista apenas com as colunas que realmente existem no DataFrame
        colunas_existentes = [
            col for col in colunas_para_remover if col in previsao_de_renda.columns
        ]

        if colunas_existentes:
            # Drop das colunas indesejadas existentes e notificação na sidebar
            previsao_de_renda = previsao_de_renda.drop(columns=colunas_existentes)
            st.sidebar.success(
                f"Variáveis indesejadas removidas do dataframe --> {colunas_existentes}"
            )

        # Seleciona somente as colunas numéricas do DataFrame (tipos number)
        numeric_dataframe = previsao_de_renda.select_dtypes(include="number")

        # Se existirem colunas numéricas, prossegue com a análise de correlação
        if not numeric_dataframe.empty:
            # Calcula a matriz de correlação das variáveis numéricas
            correlation_matrix = numeric_dataframe.corr()

        # Cria duas abas (tabs) na interface: uma para correlação, outra para gráficos
        first_tab_data_correlation, second_tab_graphics_vizualization = st.tabs(
            [
                "Visualização da correlação dos dados",
                "Análise das variáveis por meio de gráficos",
            ]
        )

        # ------------------ PRIMEIRA ABA: CORRELAÇÃO ------------------
        with first_tab_data_correlation:
            # Dentro da aba, cria um expander que pode ser aberto/fechado
            with st.expander("Correlação de variáveis numéricas", expanded=False):
                # Divide a área em duas colunas para layout paralelo
                col1, col2 = st.columns(2)  # Divide em duas colunas na tela

                # ------------------ COLUNA 1 ------------------
                with col1:
                    st.write("#### Matriz de correlação das variáveis numéricas")

                    # Faz uma cópia da matriz de correlação para exibição
                    correlation_display = correlation_matrix.copy()
                    # Substitui a diagonal por NaN para "ocultar" autocorrelações (sempre 1.0)
                    np.fill_diagonal(correlation_display.values, np.nan)
                    # Exibe a matriz estilizada com gradiente de cores
                    st.dataframe(
                        correlation_display.style.background_gradient(cmap="coolwarm")
                    )

                    # Checkbox que, se marcado, exibe pairplot e clustermap
                    if st.checkbox("Exibir pairplot do dataframe de correlação"):
                        # Divide o espaço em duas colunas para mostrar os dois gráficos lado a lado
                        col1_pairplot, col2_pairplot = st.columns(2)
                        with col1_pairplot:
                            # ---- Pairplot ----
                            # Cria um pairplot usando seaborn. O pairplot normalmente plota pares de variáveis
                            # a partir de um DataFrame "obs x variáveis". Aqui o autor passa a matriz de correlação.
                            # (No esquema original isso é o que foi feito.)
                            fig = sns.pairplot(correlation_matrix)
                            # Salva a figura em um buffer memória para exibir com st.image
                            buff = BytesIO()
                            fig.savefig(buff, format="png", bbox_inches="tight")
                            buff.seek(0)
                            st.image(buff, width=800)

                        with col2_pairplot:
                            # ---- Clustermap ----
                            # Define um mapa de cores divergente para o heatmap hierárquico
                            cmap = sns.diverging_palette(
                                h_neg=220,
                                h_pos=20,
                                as_cmap=True,
                                sep=60,
                                center="light",
                            )
                            # Cria uma máscara: aqui o código usa np.abs(correlation_matrix) < 0
                            # OBS.: np.abs(...) < 0 será sempre False (abs >= 0), então a máscara resultante é toda False.
                            #       O efeito prático é que nada será mascarado.
                            mask = np.abs(correlation_matrix) < 0

                            # Gera o clustermap (matriz com dendrogramas + heatmap)
                            fig_two = sns.clustermap(
                                correlation_matrix,
                                figsize=(10, 10),
                                center=0,
                                cmap=cmap,
                                annot=True,
                                fmt=".2f",
                                annot_kws={"size": 8},
                                mask=mask,
                            )
                            # Salva em buffer para exibir
                            buff_two = BytesIO()
                            fig_two.savefig(buff_two, format="png", bbox_inches="tight")
                            buff_two.seek(0)
                            st.image(buff_two, width=800)

                # ------------------ COLUNA 2 ------------------
                with col2:
                    st.write("#### Ranking de Correlação")

                    # Caixa de seleção para escolher a variável cujo ranking queremos ver
                    selecao_de_variavel_numerica = st.selectbox(
                        "Escolha uma variável para ver seu ranking de correlação:",
                        options=correlation_matrix.columns,
                        key="variaveis_numericas_key",
                    )

                    # Obtém a coluna de correlações relativa à variável selecionada
                    correlacoes_da_variavel = correlation_matrix[
                        selecao_de_variavel_numerica
                    ].copy()

                    # Remove a auto-correlação (a própria variável não entra no ranking)
                    correlacoes_sem_auto = correlacoes_da_variavel[
                        correlacoes_da_variavel.index != selecao_de_variavel_numerica
                    ]

                    # Monta um DataFrame com as variáveis e seus coeficientes de correlação,
                    # ordenando da maior para a menor correlação
                    hierarquia_de_correlacao = pd.DataFrame(
                        {
                            "Variável": correlacoes_sem_auto.index,
                            "Correlação": correlacoes_sem_auto.values,
                        }
                    ).sort_values("Correlação", ascending=False)

                    # Adiciona uma coluna "Posição" com o ranking numérico (1, 2, 3, ...)
                    hierarquia_de_correlacao["Posição"] = range(
                        1, len(hierarquia_de_correlacao) + 1
                    )

                    # Mostra o ranking como tabela no app
                    st.dataframe(hierarquia_de_correlacao.reset_index(drop=True))

                    # ---- Busca por posição específica ----
                    st.subheader("🔍 Posição de Variáveis Específicas")

                    # Caixa para escolher uma variável específica do ranking
                    variavel_para_buscar = st.selectbox(
                        "Buscar posição específica:",
                        options=hierarquia_de_correlacao["Variável"].tolist(),
                        key="buscar_posicao_key",
                    )

                    # Encontra a posição da variável buscada no DataFrame de ranking
                    posicao = hierarquia_de_correlacao[
                        hierarquia_de_correlacao["Variável"] == variavel_para_buscar
                    ]["Posição"].iloc[0]

                    # Encontra o valor da correlação correspondente
                    correlacao_valor = hierarquia_de_correlacao[
                        hierarquia_de_correlacao["Variável"] == variavel_para_buscar
                    ]["Correlação"].iloc[0]

                    # Exibe a posição e a correlação em um componente métrico (st.metric)
                    st.metric(
                        f"🏆 Posição de '{variavel_para_buscar}'",
                        f"{posicao}º lugar",
                        f"Correlação: {correlacao_valor:.3f}",
                    )

                    # ---- Insights simples sobre a força da correlação ----
                    st.subheader("📊 Insights")
                    if abs(correlacao_valor) > 0.7:
                        st.success("**Alta correlação** - Relação forte detectada")
                    elif abs(correlacao_valor) > 0.3:
                        st.info("**Correlação moderada** - Relação significativa")
                    else:
                        st.warning("**Baixa correlação** - Relação fraca")

        # ------------------ SEGUNDA ABA: ANÁLISE GRÁFICA ------------------
        with second_tab_graphics_vizualization:
            # Cria duas abas internas: uma para scatterplots e outra para boxplots
            aba_scatterplot, aba_boxplot = st.tabs(
                ["Visualização de scatterplot", "Visualização de boxplot"]
            )
            # ------------------ ABA SCATTERPLOT ------------------
            with aba_scatterplot:

                with st.expander(
                    "Análise de gráficos relacionados as variáveis", expanded=False
                ):

                    # Opções para eixo X: todas as colunas numéricas exceto 'renda'
                    opcoes_para_scatterplot = [
                        col for col in numeric_dataframe.columns if col != "renda"
                    ]

                    if opcoes_para_scatterplot:  # Verifica se há opções disponíveis
                        # Caixa para selecionar a variável que ficará no eixo X
                        selecao_de_variavel_para_eixo_x = st.selectbox(
                            "Selecione uma variável para visualizar a distribuição da mesma, relacionada a renda:",
                            options=opcoes_para_scatterplot,
                            key="opcoes_scatterplot_key",
                        )
                        # Divide a área em duas colunas para mostrar dois gráficos lado a lado
                        main_col1, main_col2 = st.columns(2)
                        with main_col1:

                            # Cria figura e eixo com matplotlib
                            fig, ax = plt.subplots(figsize=(10, 6))

                            # Cria scatterplot: variável selecionada vs renda
                            sns.scatterplot(
                                x=selecao_de_variavel_para_eixo_x,
                                y="renda",
                                data=previsao_de_renda,
                                alpha=0.6,
                                ax=ax,
                            )

                            # Adiciona título e rótulos ao gráfico
                            plt.title(
                                f"Renda vs {selecao_de_variavel_para_eixo_x}",
                                fontsize=14,
                                fontweight="bold",
                            )
                            plt.xlabel(selecao_de_variavel_para_eixo_x)
                            plt.ylabel("Renda")

                            # Salva a figura em um buffer (BytesIO) e exibe como imagem no Streamlit
                            buff_three = BytesIO()
                            fig.savefig(
                                buff_three, format="png", bbox_inches="tight", dpi=300
                            )
                            buff_three.seek(0)
                            st.image(buff_three, width=800)
                            # -> Observação: poderia usar st.pyplot(fig) diretamente; aqui foi usado st.image.

                            # Exibe estatísticas úteis em três colunas: correlação, R² e n de observações
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                # Calcula correlação entre a variável-X selecionada e 'renda'
                                correlacao = (
                                    previsao_de_renda[
                                        [selecao_de_variavel_para_eixo_x, "renda"]
                                    ]
                                    .corr()
                                    .iloc[0, 1]
                                )
                                st.metric("📈 Correlação", f"{correlacao:.3f}")

                            with col2:
                                # Coeficiente de determinação aproximado como correlação ao quadrado
                                coef_determinacao = correlacao**2
                                st.metric("🔍 R²", f"{coef_determinacao:.3f}")

                            with col3:
                                # Número de observações no DataFrame inteiro
                                n_observacoes = len(previsao_de_renda)
                                st.metric("📊 Observações", n_observacoes)

                        with main_col2:
                            # Cria nova cópia do DataFrame e adiciona coluna log_renda
                            dataframe_log_renda = previsao_de_renda.copy()
                            dataframe_log_renda["log_renda"] = np.log(
                                dataframe_log_renda["renda"]
                            )
                            # -> Observação: np.log em valores <= 0 produzirá -inf ou NaN;
                            #    no seu caso, se os dados estão padronizados, isso pode não ocorrer.

                            # Cria scatterplot com log_renda no eixo Y
                            fig_2, ax_2 = plt.subplots(figsize=(10, 6))
                            sns.scatterplot(
                                x=selecao_de_variavel_para_eixo_x,
                                y="log_renda",
                                data=dataframe_log_renda,
                                alpha=0.6,
                                ax=ax_2,
                            )

                            # Títulos e rótulos
                            plt.title(
                                f"Log_renda vs {selecao_de_variavel_para_eixo_x}",
                                fontsize=14,
                                fontweight="bold",
                            )
                            plt.xlabel(selecao_de_variavel_para_eixo_x)
                            plt.ylabel("Log_renda")

                            # Salva e exibe a figura com st.image
                            buff_three_log = BytesIO()
                            fig_2.savefig(
                                buff_three_log,
                                format="png",
                                bbox_inches="tight",
                                dpi=300,
                            )
                            buff_three_log.seek(0)
                            st.image(buff_three_log, width=800)

                            # Exibe métricas para o gráfico com log_renda também
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                correlacao = (
                                    dataframe_log_renda[
                                        [selecao_de_variavel_para_eixo_x, "log_renda"]
                                    ]
                                    .corr()
                                    .iloc[0, 1]
                                )
                                st.metric("📈 Correlação", f"{correlacao:.3f}")

                            with col2:
                                coef_determinacao = correlacao**2
                                st.metric("🔍 R²", f"{coef_determinacao:.3f}")

                            with col3:
                                n_observacoes = len(dataframe_log_renda)
                                st.metric("📊 Observações", n_observacoes)

                    else:
                        # Caso não haja colunas disponíveis para comparação com 'renda'
                        st.warning(
                            "Não há variáveis disponíveis para comparação com 'renda'"
                        )

            # ------------------ ABA BOXPLOT / OUTLIERS ------------------
            with aba_boxplot:
                st.write(
                    "#### Análise da existência de outliers para as variávels selecionadas"
                )
                st.info(
                    "Objetico central é análisar outliers da variável renda e se há coerência nos valores"
                )
                # Caixa de seleção para escolher a variável numérica a ser analisada no boxplot
                boxplot_selectvar_numerica = st.selectbox(
                    label="Seleção da variável numérica a ser análisada no boxplot",
                    options=numeric_dataframe.columns,
                )

                # Divide a área em duas colunas: à esquerda o boxplot, à direita as métricas/controle
                col1, col2 = st.columns(2)
                with col1:
                    # Gera o boxplot da variável selecionada
                    fig_four = plt.figure(figsize=(10, 6))
                    sns.boxplot(y=previsao_de_renda[boxplot_selectvar_numerica])
                    plt.title(f"Boxplot de {boxplot_selectvar_numerica}")

                    # Salva em buffer e exibe
                    buff_four = BytesIO()
                    fig_four.savefig(
                        buff_four, format="png", bbox_inches="tight", dpi=300
                    )
                    buff_four.seek(0)
                    st.image(buff_four, width=800)
                with col2:
                    # Calcula Q1, Q3 e IQR para definição de limites pelo método IQR
                    Q1 = previsao_de_renda[boxplot_selectvar_numerica].quantile(0.25)
                    Q3 = previsao_de_renda[boxplot_selectvar_numerica].quantile(0.75)
                    IQR = Q3 - Q1

                    limite_inferior = Q1 - 1.5 * IQR
                    limite_superior = Q3 + 1.5 * IQR

                    # Exibe estatísticas descritivas (média, mediana, Q1, Q3)
                    st.write("**📊 Estatísticas da Variável:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Média",
                            f"{previsao_de_renda[boxplot_selectvar_numerica].mean():.2f}",
                        )
                    with col2:
                        st.metric(
                            "Mediana",
                            f"{previsao_de_renda[boxplot_selectvar_numerica].median():.2f}",
                        )
                    with col3:
                        st.metric("Q1", f"{Q1:.2f}")
                    with col4:
                        st.metric("Q3", f"{Q3:.2f}")

                    # Exibe os limites calculados para definir outliers
                    st.write(
                        f"**Limites de Outliers:** Inferior = {limite_inferior:.2f}, Superior = {limite_superior:.2f}"
                    )

                    # Radio para escolher quais outliers visualizar (todos/superiores/inferiores)
                    which_outliers = st.radio(
                        "Selecione os outliers que deseja visualizar:",
                        [
                            "Todos os outliers",
                            "Somente os valores superiores",
                            "Somente os valores inferiores",
                        ],
                        horizontal=True,
                    )

                    # Detecção de outliers conforme a opção escolhida
                    if which_outliers == "Todos os outliers":
                        dataframe_outliers = previsao_de_renda[
                            (
                                previsao_de_renda[boxplot_selectvar_numerica]
                                < limite_inferior
                            )
                            | (
                                previsao_de_renda[boxplot_selectvar_numerica]
                                > limite_superior
                            )
                        ]
                        outlier_type = "superiores e inferiores"
                    elif which_outliers == "Somente os valores superiores":
                        dataframe_outliers = previsao_de_renda[
                            previsao_de_renda[boxplot_selectvar_numerica]
                            > limite_superior
                        ]
                        outlier_type = "superiores"
                    else:
                        dataframe_outliers = previsao_de_renda[
                            previsao_de_renda[boxplot_selectvar_numerica]
                            < limite_inferior
                        ]
                        outlier_type = "inferiores"

                    # Calcula totais e percentual de outliers
                    total_observacoes = len(previsao_de_renda)
                    total_outliers = len(dataframe_outliers)
                    percentual_outliers = (total_outliers / total_observacoes) * 100

                    # Exibe resultados
                    st.write(f"**🔍 Resultados da Análise:**")
                    st.write(f"- Total de observações: {total_observacoes}")
                    st.write(f"- Outliers {outlier_type} encontrados: {total_outliers}")
                    st.write(f"- Percentual de outliers: {percentual_outliers:.2f}%")

                    # Avaliação qualitativa do percentual
                    if percentual_outliers > 5:
                        st.warning(
                            "⚠️ **Alta porcentagem de outliers** - Pode indicar necessidade de tratamento"
                        )
                    elif percentual_outliers > 0:
                        st.success("✅ **Proporção de outliers dentro do esperado**")
                    else:
                        st.info("📊 **Nenhum outlier detectado** com o método IQR")

                    # Se foram encontrados outliers, fornece abas com detalhes e gráficos
                    if not dataframe_outliers.empty:
                        # Três abas: tabela, gráfico comparativo e análise detalhada
                        tab1, tab2, tab3 = st.tabs(
                            [
                                "📋 Dados dos Outliers",
                                "📈 Visualização Gráfica",
                                "🔍 Análise Detalhada",
                            ]
                        )

                        with tab1:
                            # Mostra os dados dos outliers em tabela
                            st.dataframe(dataframe_outliers)

                            # Botão para baixar os outliers como CSV
                            csv = dataframe_outliers.to_csv(index=False)
                            st.download_button(
                                label="📥 Download dos Outliers (CSV)",
                                data=csv,
                                file_name=f"outliers_{boxplot_selectvar_numerica}.csv",
                                mime="text/csv",
                            )

                        with tab2:
                            # Gráfico comparativo: lado a lado (boxplot) e histograma comparativo
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                            # Define "dados normais" filtrando pelo intervalo IQR (não outliers)
                            dados_normais = previsao_de_renda[
                                (
                                    previsao_de_renda[boxplot_selectvar_numerica]
                                    >= limite_inferior
                                )
                                & (
                                    previsao_de_renda[boxplot_selectvar_numerica]
                                    <= limite_superior
                                )
                            ]

                            # Boxplot comparativo entre dados normais e outliers
                            ax1.boxplot(
                                [
                                    dados_normais[boxplot_selectvar_numerica],
                                    dataframe_outliers[boxplot_selectvar_numerica],
                                ],
                                labels=["Dados Normais", "Outliers"],
                            )
                            ax1.set_title(f"Comparação: Dados Normais vs Outliers")
                            ax1.set_ylabel(boxplot_selectvar_numerica)

                            # Histograma comparativo para visualizar distribuição
                            ax2.hist(
                                dados_normais[boxplot_selectvar_numerica],
                                alpha=0.7,
                                label="Normais",
                                bins=30,
                            )
                            ax2.hist(
                                dataframe_outliers[boxplot_selectvar_numerica],
                                alpha=0.7,
                                label="Outliers",
                                bins=10,
                            )
                            ax2.set_title("Distribuição Comparativa")
                            ax2.legend()

                            # Exibe o gráfico com st.pyplot
                            st.pyplot(fig)

                        with tab3:
                            # Mostra estatísticas descritivas dos valores considerados outliers
                            st.write("**Estatísticas Descritivas dos Outliers:**")
                            st.dataframe(
                                dataframe_outliers[
                                    boxplot_selectvar_numerica
                                ].describe()
                            )

                            # Se houver pelo menos 2 outliers e a coluna 'renda' existir,
                            # compara a correlação entre a variável e renda nos outliers vs geral
                            if (
                                len(dataframe_outliers) > 1
                                and "renda" in previsao_de_renda.columns
                            ):
                                correlacao_outliers = (
                                    dataframe_outliers[
                                        [boxplot_selectvar_numerica, "renda"]
                                    ]
                                    .corr()
                                    .iloc[0, 1]
                                )
                                correlacao_geral = (
                                    previsao_de_renda[
                                        [boxplot_selectvar_numerica, "renda"]
                                    ]
                                    .corr()
                                    .iloc[0, 1]
                                )

                                st.write("**📈 Comparação de Correlações:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Correlação Geral", f"{correlacao_geral:.3f}"
                                    )
                                with col2:
                                    st.metric(
                                        "Correlação nos Outliers",
                                        f"{correlacao_outliers:.3f}",
                                    )
                    else:
                        # Caso nenhum outlier tenha sido identificado, exibe mensagem de sucesso
                        st.success(
                            "🎉 Nenhum outlier encontrado com os critérios atuais!"
                        )


# ------------------------------------------------------
# PONTO DE ENTRADA
# ------------------------------------------------------
if __name__ == "__main__":
    main()  # Executa a função principal ao rodar o script
