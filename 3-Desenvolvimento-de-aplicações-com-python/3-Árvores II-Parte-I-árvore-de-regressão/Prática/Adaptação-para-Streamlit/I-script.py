import pydotplus as pydot  # usado para manipulação de grafos (opcional no fluxo atual, mas importado)
import pandas as pd  # manipulação de dataframes
import graphviz  # ferramenta para renderizar grafos (importada mas não explicitamente usada depois)
import numpy as np  # operações numéricas e matriciais
import seaborn as sns  # visualizações estatísticas
import streamlit as st  # interface web interativa
import matplotlib.pyplot as plt  # plotagem com matplotlib
from io import BytesIO  # buffer em memória para salvar imagens e exibir no Streamlit
from sklearn.model_selection import train_test_split  # divisão treino/validação/teste
from sklearn.tree import DecisionTreeRegressor  # modelo de árvore de regressão
from sklearn import tree  # utilidades para árvores
from sklearn.tree import plot_tree  # função para desenhar a árvore com matplotlib


# ------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ------------------------------------------------------
st.set_page_config(
    page_title="Modelando com scikit-learn",  # Título exibido na aba do navegador
    layout="wide",  # Usa largura total da janela
    initial_sidebar_state="expanded",  # sidebar aberta por padrão
    page_icon="varig_icon.png",  # ícone da aplicação (arquivo esperado localmente)
)


# ------------------------------------------------------
# FUNÇÃO PARA CARREGAR DADOS
# ------------------------------------------------------
def load_data(uploaded_file):
    """
    Lê um arquivo CSV enviado via widget do Streamlit e retorna um DataFrame.
    - Em caso de erro na leitura, exibe mensagem na interface (st.error).
    - Mantém comportamento simples/hardcoded: usa pd.read_csv.
    """
    try:
        data = pd.read_csv(uploaded_file)  # Leitura do CSV para DataFrame
        return data
    except Exception as e:
        # Se ocorrer erro (e.g., arquivo mal formatado), mostra mensagem de erro
        st.error(f"Não foi possivel carregar o dado selecionado. {e}")


# ------------------------------------------------------
# FUNÇÃO PRINCIPAL DO APP
# ------------------------------------------------------
def main():
    # Layout do cabeçalho: três colunas com proporções customizadas
    titulo_col_1, titulo_col_2, titulo_col_3 = st.columns([1.5, 4, 1])
    # Título centralizado (na coluna do meio)
    titulo_col_2.title("Árvores de classificação com `scikit-learn`")

    # Widget de upload na sidebar: pede um CSV (housing.csv no repositório)
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Faça o upload do dataframe `housing.csv`, disponibilizado no repositório",
            type="csv",
        )

    # Se o usuário carregou um arquivo, inicia o fluxo de pré-processamento e modelagem
    if uploaded_file is not None:
        data = load_data(uploaded_file)  # Lê os dados com a função acima

        # --------------------------------------------------
        # EXPANDER: Adequação do dataset para modelagem
        # --------------------------------------------------
        with st.expander(
            "Adequação do dataset para modelagem com `scikit-learn`", expanded=True
        ):
            # Seção: tratamento de missings (valores nulos)
            st.write("#### Tratamento de `missings`")
            coluna_nulos_originais, coluna_nulos_tratados = st.columns(2)

            # Mostra a contagem de valores nulos por coluna (antes do tratamento)
            with coluna_nulos_originais:
                st.info("**Contagem de valores nulos por variável do dataset:**")
                st.dataframe(data.isna().sum())

            # Mostra a contagem após a exclusão de linhas com nulos (dropna)
            with coluna_nulos_tratados:
                st.success(
                    "**Todos os nulos presentes na primeira visualização foram excluídos**"
                )
                st.dataframe(data.dropna().isna().sum())

            st.divider()  # divisor visual

            # Seção: análise dos tipos de coluna e preparação para conversão de categóricas
            st.write("#### Remoção de variáveis do tipo `object`")
            col_1_secao_2, col_2_secao_2 = st.columns(2)

            with col_1_secao_2:
                # Monta um dicionário com metadados básicos por coluna:
                # nome da coluna, tipo, quantidade de não-nulos e número de valores únicos
                info_data = {
                    "Coluna": data.columns,
                    "Tipo": data.dtypes,
                    "Não Nulos": data.dropna().count(),
                    "Valores Únicos": [data[col].nunique() for col in data.columns],
                }
                st.write("##### Análise dos tipos")
                # Exibe a tabela com as infos (hide_index=True remove índice visual)
                st.dataframe(info_data, hide_index=True)

                # Identifica colunas que são objetos e possuem mais de 2 valores únicos
                # (provavelmente necessitam de conversão em dummies / one-hot)
                colunas_para_converter = []
                for col in data.columns:
                    if (data[col].dtype == "object") & (data[col].nunique() > 2):
                        colunas_para_converter.append(col)

                # Mostra aviso dependendo da quantidade de colunas que precisam conversão
                if len(colunas_para_converter) == 1:
                    st.warning(f"Apensa uma coluna precisa convertida")
                    st.write("##### Colunas que necessitam de tratamento:")
                    st.write(f"`{colunas_para_converter[0]}`")
                else:
                    st.warning(
                        f"Um total de {len(colunas_para_converter)} precisam ser convertidas para dummies"
                    )
                    st.write("##### Colunas que necessitam de tratamento:")
                    st.write(f"`{colunas_para_converter}`")

            # Segunda coluna da seção: demonstra o resultado da conversão para dummies
            with col_2_secao_2:
                st.write("##### Resultado da conversão em dummies")
                # Cria dummies (variáveis dummy/one-hot) e remove a primeira categoria (drop_first=True)
                data_dummies = pd.get_dummies(data=data, drop_first=True)
                st.dataframe(data_dummies)

            # Aplica limpeza definitiva: remove linhas com nulos (inplace)
            data.dropna(inplace=True)
            # Recria data_dummies com dados sem nulos
            data_dummies = pd.get_dummies(data=data, drop_first=True)

            # Separa features (x) e target (y); target esperada: 'median_house_value'
            x = data_dummies.drop(columns=["median_house_value"])
            y = data_dummies["median_house_value"]

        st.divider()  # separador visual

        # --------------------------------------------------
        # EXPANDER: Entendendo importância / redundância das variáveis
        # --------------------------------------------------
        with st.expander("Entendendo a importância de cada variável", expanded=True):
            st.write("#### Redundância entre as variáveis")
            # Calcula matriz de correlação entre as features X
            correlation_matrix = x.corr()
            col1, col2 = st.columns(2)

            # Coluna 1: clustermap (heatmap hierárquico) da matriz de correlação
            with col1:
                cmap = sns.diverging_palette(
                    h_neg=220, h_pos=20, as_cmap=True, sep=60, center="light"
                )
                # Máscara definida com condição que resulta sempre em False (np.abs(...) < 0)
                # (o código mantém essa máscara como está; na prática ela não mascara nada)
                mask = np.abs(correlation_matrix) < 0

                # Gera clustermap (heatmap com dendrograma) usando seaborn
                grid = sns.clustermap(
                    correlation_matrix,
                    figsize=(10, 10),
                    center=0,
                    cmap=cmap,
                    annot=True,
                    fmt=".2f",
                    annot_kws={"size": 8},
                    mask=mask,
                )

                # Salva o gráfico em memória e exibe no Streamlit como imagem
                buff = BytesIO()
                grid.savefig(buff, format="png", bbox_inches="tight")
                buff.seek(0)
                st.image(buff, width=800)

            # Coluna 2: cálculo e exibição das top correlações
            with col2:
                # 'unstack' transforma a matriz em série (pares de variáveis -> correlação)
                corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)

                # Remove entradas onde a variável é a mesma (auto-correlação)
                unique_corr_pairs = corr_pairs[
                    corr_pairs.index.get_level_values(0)
                    != corr_pairs.index.get_level_values(1)
                ]

                # Função auxiliar para remover pares duplicados (A,B e B,A)
                def remove_duplicate_correlations(series):
                    seen = set()
                    result = []
                    for (var1, var2), value in series.items():
                        pair = frozenset([var1, var2])  # par sem ordem
                        if pair not in seen:
                            seen.add(pair)
                            result.append(((var1, var2), value))
                    # Retorna uma Series onde índice é tupla (var1,var2) e valor é correlação
                    return pd.Series(dict(result))

                # Aplica a função para deixar apenas uma entrada por par de variáveis
                unique_corr_pairs = remove_duplicate_correlations(unique_corr_pairs)

                # Seleciona os 5 pares com maior correlação (top 5)
                top_5_correlacoes = unique_corr_pairs.head(5)

                # Monta DataFrame amigável para exibir os top 5 (posição, variáveis, valor e interpretação)
                st.write("**Top 5 Correlações Positivas Mais Fortes:**")
                top_5_df = pd.DataFrame(
                    [
                        {
                            "Posição": i + 1,
                            "Variável 1": pair[0],
                            "Variável 2": pair[1],
                            "Correlação": f"{corr:.3f}",
                            "Interpretação": (
                                "Forte"
                                if corr > 0.7
                                else "Moderada" if corr > 0.3 else "Fraca"
                            ),
                        }
                        for i, (pair, corr) in enumerate(top_5_correlacoes.items())
                    ]
                )

                # Mostra a tabela com largura ajustada ao container
                st.dataframe(top_5_df, use_container_width=True)

        st.divider()  # separador visual

        # --------------------------------------------------
        # DIVISÃO DO CONJUNTO: treino / validação / teste
        # --------------------------------------------------
        # 1) primeiro split: separa 20% para teste final
        X_, X_test, y_, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
        # 2) segundo split: divide o restante em treino e validação (aqui 25% -> validação)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_, y_, test_size=0.25, random_state=100
        )

        # Exibe informações sobre as divisões em um expander
        with st.expander("Divisão da base de dados", expanded=True):
            st.write("#### Divisão do conjunto de dados em treino, validação e teste")
            coluna_divisao_dataset_1, coluna_divisao_dataset_2 = st.columns(2)
            with coluna_divisao_dataset_1:
                # Mostra o número de linhas em cada subset
                st.metric("Número de linhas do dataset de treino:", X_train.shape[0])
                st.metric("Número de linhas do dataset de validação:", X_valid.shape[0])
                st.metric("Número de linhas do dataset de teste:", X_test.shape[0])
            with coluna_divisao_dataset_2:
                # Calcula e mostra as proporções relativas ao dataset original
                proporcao_treino = round((X_train.shape[0] / x.shape[0]) * 100, 3)
                st.metric(
                    "Proporção do dataset original para o treino:", proporcao_treino
                )
                proporcao_validacao = round((X_valid.shape[0] / x.shape[0]) * 100, 3)
                st.metric(
                    "Proporção do dataset original para a validação:",
                    proporcao_validacao,
                )
                proporcao_teste = round((X_test.shape[0] / x.shape[0]) * 100, 3)
                st.metric(
                    "Proporção do dataset original para o treino:", proporcao_teste
                )

        st.divider()  # separador visual

        # --------------------------------------------------
        # TREINAMENTO: cria e treina duas árvores de regressão com profundidades diferentes
        # --------------------------------------------------
        regr_1 = DecisionTreeRegressor(
            max_depth=8
        )  # árvore mais profunda (mais complexa)
        regr_1.fit(X_train, y_train)  # ajuste nos dados de treino

        regr_2 = DecisionTreeRegressor(max_depth=2)  # árvore rasa (mais simples)
        regr_2.fit(X_train, y_train)  # ajuste nos dados de treino

        # Exibe status e métricas de performance dentro de um expander
        with st.expander("Treinamento dos modelos", expanded=True):
            st.write("#### Treinamento e avaliação de duas árvores de decisão:")
            # Verifica (de forma simples) se os modelos foram instanciados
            if (regr_1) and (regr_2) is not None:
                st.success("Modelos criados com sucesso")

            # .score em regressor retorna R^2 por padrão
            mse1_train = regr_1.score(X_train, y_train)
            mse2_train = regr_2.score(X_train, y_train)
            mse1_test = regr_1.score(X_test, y_test)
            mse2_test = regr_2.score(X_test, y_test)

            template = (
                "O R-Quadrado dos treinos para árvore com profundidade={0} é: {1:.2f}"
            )
            col1, col2 = st.columns(2)

            # Exibe métricas (R²) de treino e teste formatadas como porcentagem aproximada (multiplica por 100)
            with col1:
                st.write("##### Performance em treinos")
                st.metric(
                    "Árvore de regressão com 8 de profundidade",
                    round(mse1_train, 2) * 100,
                )
                st.metric(
                    "Árvore de regressão com 2 de profundidade",
                    round(mse2_train, 3) * 100,
                )
            with col2:
                st.write("##### Performance em testes")
                st.metric(
                    "Árvore de regressão com 8 de profundidade",
                    round(mse1_test, 2) * 100,
                )
                st.metric(
                    "Árvore de regressão com 2 de profundidade",
                    round(mse2_test, 3) * 100,
                )

        st.divider()  # separador visual

        # --------------------------------------------------
        # VISUALIZAÇÃO: diagrama da árvore escolhida
        # --------------------------------------------------
        with st.expander("Visualização de diagrama dos modelos:"):
            # Radio para o usuário escolher qual diagrama (profundidade) deseja visualizar
            radio_diagrama_arvore = st.radio(
                label="Selecione o modelo que você deseja visualizar o diagrama",
                options=[
                    "Árvores com 2 de profundidade",
                    "Árvore com 8 de profundidade",
                ],
                horizontal=True,
                key="selecao_diagrama_arvore_key",
            )

            # Função auxiliar: retorna o nome da feature com maior importância segundo o modelo
            def get_most_important_feature(model, feature_names):
                importances = model.feature_importances_
                most_important_idx = importances.argmax()  # índice da maior importância
                return feature_names[most_important_idx]

            # Determina o modelo a ser exibido conforme a escolha do usuário
            if radio_diagrama_arvore == "Árvores com 2 de profundidade":
                # Nota: lógica aqui usa regr_1 se sua profundidade for 2, senão usa regr_2
                model = regr_1 if regr_1.get_depth() == 2 else regr_2
                title_depth = "2"
                font_size = 12
            else:
                # Similarmente, para profundidade 8 tenta regr_1, senão regr_2
                model = regr_1 if regr_1.get_depth() == 8 else regr_2
                title_depth = "8"
                font_size = 8

            # Identifica a variável mais importante no modelo selecionado
            important_feature = get_most_important_feature(model, x.columns)

            # Mostra informação textual destacando a variável mais importante
            st.info(
                f"🔍 **Variável mais importante para esta árvore:** **'{important_feature}'**"
            )

            # Fecha figuras matplotlib abertas (boa prática antes de criar nova figura)
            plt.close("all")
            fig, ax = plt.subplots(figsize=(20, 10))

            # Desenha a árvore com plot_tree (matplotlib), usando nomes das features e preenchimento por classe/região
            plot_tree(
                model,
                feature_names=x.columns,
                filled=True,
                ax=ax,
                fontsize=font_size,
            )

            # Título da figura incluindo profundidade e variável mais importante
            plt.title(
                f"Árvore com Profundidade {title_depth}\nVariável mais importante: {important_feature}",
                fontsize=16,
                pad=20,
            )

            plt.tight_layout()  # ajusta layout para evitar cortes
            # Salva em buffer e exibe no Streamlit
            buff_diagram = BytesIO()
            plt.savefig(buff_diagram, format="png", dpi=300, bbox_inches="tight")
            buff_diagram.seek(0)
            st.image(buff_diagram)
            plt.close(fig)  # fecha a figura explicitamente


# ------------------------------------------------------
# PONTO DE ENTRADA
# ------------------------------------------------------
if __name__ == "__main__":
    main()  # Executa a função principal ao rodar o script
