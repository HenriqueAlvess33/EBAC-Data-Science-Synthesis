# Importações necessárias
from io import BytesIO
import pandas as pd
import time
import streamlit as st
import numpy as np
from concurrent.futures import ThreadPoolExecutor  # (não utilizado ainda)
from io import StringIO  # (não utilizado ainda)
import threading  # (não utilizado ainda)
from pandas.api.types import CategoricalDtype
import altair as alt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# ------------------------------------------------------
# CONFIGURAÇÃO INICIAL DA PÁGINA STREAMLIT
# ------------------------------------------------------
st.set_page_config(
    page_title="Preparando a base",  # Título da aba do navegador
    layout="wide",  # Layout expandido
    initial_sidebar_state="expanded",  # Sidebar aberta por padrão
    page_icon="varig_icon.png",  # Ícone da página
)

# ------------------------------------------------------
# INICIALIZAÇÃO DE VARIÁVEIS NA SESSÃO STREAMLIT
# ------------------------------------------------------
# Isso garante que variáveis de estado estejam sempre disponíveis,
# evitando erros de "variável não definida".for key, value in session_defaults.items():

session_defaults = {
    "df": None,
    "show_success": None,
    "conversion_applied": None,
    "data_loaded": None,
    "df_converted": None,
    "df_atividade_2": None,
    "data_ready": None,
    "data_loaded_2": False,
    "x_train": None,
    "y_train": None,
    "x_test": None,
    "y_test": None,
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ------------------------------------------------------
# FUNÇÃO PRINCIPAL DA APLICAÇÃO
# ------------------------------------------------------
def main():
    # Cria container vazio para mensagens de sucesso temporárias
    success_container = st.empty()

    # Sidebar com seleção de atividade
    with st.sidebar:
        selecao_de_atividade = st.radio(
            "Neste módulo foram realizadas 2 atividades, selecione qual delas você deseja ver",
            ["Atividade 1", "Atividade 2"],
            key="selecao_de_atividade_key",
            horizontal=True,
        )
        st.info(
            "O conjunto de dados utilizado na segunda atividade é o resultante da primeira, "
            "ou seja, tendo passado pelo tratamento de dummies e estando apto a trabalhar com scikit-learn"
        )

    # ------------------------------------------------------
    # ATIVIDADE 1 - CARREGAMENTO E EXPLORAÇÃO DE DADOS
    # ------------------------------------------------------
    if selecao_de_atividade == "Atividade 1":
        # Caso os dados ainda não tenham sido carregados
        if not st.session_state.data_loaded:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Simulação de carregamento para experiência do usuário
            for i in range(3):
                time.sleep(0.5)
                progress_bar.progress((i + 1) * 33)
                status_text.text(f"Processando... {((i+1)*33)}% completo")

            try:
                # Leitura do dataset real (modifique o caminho se necessário)
                df = pd.read_csv("./input/demo01.csv")

                # Armazena no estado da sessão
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.show_success = True

            except Exception as e:
                st.error(f"Erro ao carregar dados: {e}")
                return

            # Limpa barra de progresso e texto
            progress_bar.empty()
            status_text.empty()

        # Exibe mensagem de sucesso (se habilitada)
        if st.session_state.show_success:
            success_container.success("Dataframe carregado com sucesso!")
            time.sleep(0.5)  # Mostra por 2 segundos
            success_container.empty()
            st.session_state.show_success = False

        # ------------------------------------------------------
        # TÍTULO PERSONALIZADO (HTML + CSS inline)
        # ------------------------------------------------------
        st.markdown(
            """
            <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">
            <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 2.5em;'>
                <strong> Preparativos de uma base para construção de uma árvore de classificação</strong>
            </h1>
            """,
            unsafe_allow_html=True,
        )

        # Só continua se os dados foram carregados corretamente
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df

            # ------------------------------------------------------
            # VISÃO GERAL DO DATASET
            # ------------------------------------------------------
            with st.expander("📊 VISÃO GERAL DO DATASET", expanded=False):
                st.subheader("Informações do DataFrame")

                # Divide em 4 colunas de resumo
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

                with col1:
                    st.metric("Total de linhas:", df.shape[0])
                with col2:
                    st.metric("Total de colunas:", df.shape[1])
                with col3:
                    st.metric("Valores ausentes:", df.isna().sum().sum())
                with col4:
                    st.metric(
                        "Memória Aproximada:",
                        f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB",
                    )

            st.divider()

            # ------------------------------------------------------
            # ABAS DE EXPLORAÇÃO
            # ------------------------------------------------------
            tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs(
                [
                    "🏗️ ESTRUTURA",
                    "📈 ESTATÍSTICAS",
                    "👀 AMOSTRA",
                    "🎯 VARIÁVEL TARGET",
                    "⚙️ PREPARAÇÃO PARA MODELO",
                ]
            )

            # TAB 1: Estrutura de metadados
            with tab_1:
                st.subheader("Estrutura Metadados")
                col1, col2 = st.columns([1, 1])

                with col1:
                    # Cria DataFrame com informações coluna a coluna
                    info_data = {
                        "Colunas": df.columns,
                        "Tipos de dados": df.dtypes.values,
                        "Não Nulos": df.count().values,
                        "Valores Faltantes": df.isnull().sum().values,
                        "% Faltantes": (df.isnull().sum() / len(df) * 100).round(2),
                    }
                    info_df = pd.DataFrame(info_data)
                    st.dataframe(info_df)

                with col2:
                    st.subheader("Resumo por tipo")
                    type_counts = df.dtypes.value_counts()
                    for dtype, count in type_counts.items():
                        st.write(f"**{dtype}**: {count} colunas")

                    st.subheader("Ações recomendadas:")
                    if df.isnull().sum().sum() > 0:
                        st.warning("⚠️Há valores ausentes no dataframe")
                        st.write("Considere a inputação ou remoção")

                    # Detecta colunas com alta cardinalidade (muitas categorias)
                    high_cardinality = [
                        col
                        for col in df.select_dtypes(include=["object"]).columns
                        if df[col].nunique() > 50
                    ]
                    if high_cardinality:
                        st.warning("⚠️ Alta cardinalidade detectada")
                        st.write(f"Colunas: {', '.join(high_cardinality)}")

            # TAB 2: Estatísticas descritivas
            with tab_2:
                st.subheader("Análises Estatísticas")
                tipo_analise = st.radio(
                    "Tipo de análise:",
                    ["Numéricas", "Categóricas", "Todas"],
                    horizontal=True,
                    key="tipo_analise_key",
                )

                if tipo_analise == "Numéricas":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(
                            df[numeric_cols].describe(), use_container_width=True
                        )
                    else:
                        st.info("Nenhuma coluna numérica encontrada")

                elif tipo_analise == "Categóricas":
                    categorical_cols = df.select_dtypes(
                        include=["object", "category"]
                    ).columns
                    if len(categorical_cols) > 0:
                        for col in categorical_cols:
                            with st.expander(f"{col}"):
                                counts = df[col].value_counts()
                                st.dataframe(counts)
                                st.bar_chart(counts)
                    else:
                        st.info("Nenhuma coluna categórica foi encontrada")

                elif tipo_analise == "Todas":
                    st.dataframe(df.describe(include=("all")), use_container_width=True)

            # TAB 3: Amostragem
            with tab_3:
                sample_tabs = st.tabs(
                    ["Primeiras linhas", "Últimas linhas", "Amostragem aleatória"]
                )
                with sample_tabs[0]:
                    st.dataframe(
                        df.head(10), use_container_width=True, key="df_head_key"
                    )
                with sample_tabs[1]:
                    st.dataframe(
                        df.tail(10), use_container_width=True, key="df_tail_key"
                    )
                with sample_tabs[2]:
                    sample_size = st.slider(
                        "Tamanho da amostra", 5, 50, 10, key="sample_size_key"
                    )
                    st.dataframe(
                        df.sample(sample_size),
                        use_container_width=True,
                        key="df_amostragem_key",
                    )

            # TAB 4: Análise da variável target
            with tab_4:
                st.subheader("Análise da variável target")
                target = "mau"  # variável alvo pré-definida
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.write("**Distribuição:**")
                    count_target = df[target].value_counts()
                    st.dataframe(count_target, key="distribuicao_key")
                    st.bar_chart(count_target)
                with col2:
                    st.write("**Proporções:**")
                    st.dataframe(
                        (df[target].value_counts(normalize=True) * 100).round(2),
                        key="proporcoes_key",
                    )

                st.divider()
                st.write("#### Discriminação da variável target")
                variavel_para_comparacao = st.selectbox(
                    "Selecione a variável para comparação",
                    [x for x in df.columns.to_list() if x != "mau"],
                )

                # Cria tabela cruzada normalizada (%)
                tabela_cruzada_pct = (
                    pd.crosstab(df[variavel_para_comparacao], df["mau"], normalize=True)
                    * 100
                )

                # Reorganiza para visualização em gráfico
                tabela_cruzada_vis = tabela_cruzada_pct.reset_index().melt(
                    id_vars=variavel_para_comparacao,
                    var_name="Target",
                    value_name="Proporção",
                )

                chart = (
                    alt.Chart(tabela_cruzada_vis)
                    .mark_bar()
                    .encode(
                        x=alt.X(variavel_para_comparacao, title="Categoria"),
                        y=alt.Y("Proporção", title="Proporção (%)"),
                        color="Target",
                    )
                )

                coluna_esquerda, coluna_direita = st.columns(2)
                with coluna_esquerda:
                    st.dataframe(tabela_cruzada_pct)
                with coluna_direita:
                    st.altair_chart(chart, use_container_width=True)

            # TAB 5: Conversão para dummies (pré-processamento)
            with tab_5:
                st.subheader("Conversão de variáveis categóricas para dummies")

                categorical_columns = df.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()
                columns_for_dummies = [
                    col for col in categorical_columns if df[col].nunique() > 2
                ]

                if not columns_for_dummies:
                    st.success(
                        "✅ Nenhuma variável categórica com mais de 2 categorias encontrada!"
                    )
                    st.info(
                        "Variáveis com 2 categorias podem ser convertidas para 0/1 diretamente"
                    )
                else:
                    # Exibe colunas candidatas a dummies
                    st.write("### Variáveis identificadas para conversão")
                    dict_dummie_columns = []
                    for col in columns_for_dummies:
                        dict_dummie_columns.append(
                            {
                                "Coluna": col,
                                "Número de categorias": df[col].nunique(),
                                "Categorias": ", ".join(map(str, df[col].unique()[:3]))
                                + ("..." if df[col].nunique() > 3 else ""),
                            }
                        )
                    visualizacao_df = pd.DataFrame(dict_dummie_columns)
                    st.dataframe(
                        visualizacao_df,
                        hide_index=True,
                        key="dummies_visualization_key",
                    )

                    st.divider()
                    st.write("### 🎛️ Controles de conversão")

                    # Opções de conversão
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        opcoes_de_selecao = df.select_dtypes(
                            include=["object", "category"]
                        ).columns.to_list()
                        colunas_selecionadas = st.multiselect(
                            "Selecione as variáveis para converter:",
                            options=opcoes_de_selecao,
                            default=opcoes_de_selecao,
                            help="Selecione quais variáveis deseja transformar em dummies",
                            key="selected_columns_multiselect_key",
                        )
                    with col2:
                        drop_first_strategy = st.selectbox(
                            "Estratégia de remoção:",
                            options=[
                                "Manter Todas",
                                "Remover primeira categoria",
                                "Remover última categoria",
                            ],
                            key="drop_first_strategy_key",
                        )

                    # Pré-visualização da transformação
                    if colunas_selecionadas:
                        st.divider()
                        st.write("#### 👁️ Pré-visualização da Transformação")

                        for i, coluna in enumerate(colunas_selecionadas):
                            expander = st.expander(
                                f"🔍 **{coluna}** ({df[coluna].nunique()} categorias)"
                            )
                            with expander:
                                col_left, col_right = st.columns(2)

                                # Coluna da esquerda → valores originais
                                with col_left:
                                    st.write("**Valores Originais**")
                                    contagem = df[coluna].value_counts()
                                    contagem_df = pd.DataFrame(
                                        {
                                            "Categoria": contagem.index,
                                            "Contagem": contagem.values,
                                            "Percentual": (
                                                contagem.values / len(df) * 100
                                            ).round(1),
                                        }
                                    )
                                    st.dataframe(
                                        contagem_df,
                                        hide_index=True,
                                        key=f"original_values_{coluna}",
                                    )
                                    categorias = df[coluna].unique()

                                    # Mostra categoria de referência (se removida)
                                    if drop_first_strategy != "Manter Todas":
                                        if (
                                            drop_first_strategy
                                            == "Remover primeira categoria"
                                        ):
                                            ref_category = categorias[0]
                                        else:
                                            ref_category = categorias[-1]
                                        st.info(
                                            f"🔸 Categoria de referência **{ref_category}**"
                                        )

                                # Coluna da direita → após conversão
                                with col_right:
                                    st.write("**Após conversão (Dummies)**")
                                    drop_first = drop_first_strategy != "Manter Todas"
                                    df[coluna] = df[coluna].astype(
                                        CategoricalDtype(categories=categorias)
                                    )

                                    dummies_sample = pd.get_dummies(
                                        df[[coluna]],
                                        prefix=coluna,
                                        drop_first=drop_first,
                                    ).head()

                                    if dummies_sample.shape[1] > 0:
                                        st.dataframe(
                                            dummies_sample,
                                            use_container_width=True,
                                            key=f"dummies_preview_{coluna}",
                                        )
                                        st.write(
                                            f"↳ **{dummies_sample.shape[1]} novas colunas** criadas"
                                        )

                                        if drop_first:
                                            success_container.success(
                                                "✅ Multicolinearidade evitada!"
                                            )
                                        else:
                                            st.warning("⚠️ Possível multicolinearidade!")
                                    else:
                                        st.warning(
                                            "Nenhuma coluna dummy criada (apenas 1 categoria)"
                                        )

                                    # Mapping das categorias para dummies
                                    st.write("**Mapping das categorias:**")
                                    for i, cat in enumerate(categorias):
                                        if (
                                            drop_first
                                            and i == 0
                                            and drop_first_strategy
                                            == "Remover primeira categoria"
                                        ):
                                            st.write(f"- {cat} → (referência, omitida)")
                                        elif (
                                            drop_first
                                            and i == len(categorias) - 1
                                            and drop_first_strategy
                                            == "Remover última categoria"
                                        ):
                                            st.write(f"- {cat} → (referência, omitida)")
                                        else:
                                            dummy_col_name = f"{coluna}_{cat}"
                                            st.write(f"- {cat} → {dummy_col_name} = 1")

                        st.divider()
                        st.write("#### ⚡ Aplicar Conversão")

                        # Botão para aplicar conversão definitiva
                        if st.button(
                            "🔄 Converter Variáveis Selecionadas",
                            type="primary",
                            key="apply_conversion_button_key",
                        ):
                            df_converted = df.copy()
                            drop_first = drop_first_strategy != "Manter Todas"

                            for coluna in colunas_selecionadas:
                                dummies = pd.get_dummies(
                                    df_converted[coluna],
                                    prefix=coluna,
                                    drop_first=drop_first,
                                )
                                df_converted = df_converted.drop(coluna, axis=1)
                                df_converted = pd.concat(
                                    [df_converted, dummies], axis=1
                                )

                            st.session_state.df_converted = df_converted
                            st.session_state.conversion_applied = True

                            st.success("✅ Conversão aplicada com sucesso!")
                            st.info(
                                f"Dataset Atualizado: {df_converted.shape[1]} colunas totais"
                            )

                            new_columns = [
                                col
                                for col in df_converted.columns
                                if col not in df.columns
                            ]
                            st.write(f"**{len(new_columns)} novas colunas criadas:**")
                            st.write(
                                ", ".join(new_columns[:8])
                                + ("..." if len(new_columns) > 8 else "")
                            )

                    else:
                        st.warning("⚠️ Selecione pelo menos uma variável para converter")

                    # Se conversão já foi aplicada → opções de download
                    if st.session_state.conversion_applied:
                        st.divider()
                        st.write("#### Opções de Download do dataframe convertido")
                        selecao_formato = st.selectbox(
                            "Selecione o formato para Download:",
                            ["CSV", "Excel", "JSON"],
                        )

                        if selecao_formato == "CSV":
                            arquivo = st.session_state.df_converted.to_csv(
                                index=False
                            ).encode("utf-8")
                            st.download_button(
                                "📥 Baixar CSV",
                                data=arquivo,
                                file_name="dados.csv",
                                mime="text/csv",
                            )
                        elif selecao_formato == "Excel":
                            buffer = BytesIO()
                            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                                st.session_state.df_converted.to_excel(
                                    writer, index=False, sheet_name="Planilha1"
                                )
                            arquivo = buffer.getvalue()
                            st.download_button(
                                "📥 Baixar Excel",
                                data=arquivo,
                                file_name="data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml",
                            )

    # Se o usuário selecionar a "Atividade 2" no menu de seleção
    if selecao_de_atividade == "Atividade 2":
        aba_1_verificacao_dos_dados, aba_2_construcao_e_visualizacao_do_modelo = (
            st.tabs(["TRATAMENTO FINAL DOS DADOS", "CRIAÇÃO E VISUALIZAÇÃO DO MODELO"])
        )
        with aba_1_verificacao_dos_dados:
            st.markdown("## 📊 Atividade 2 - Preparação para Modelagem")

            # Caso o DataFrame da Atividade 2 ainda não esteja carregado
            # mas o DataFrame da Atividade 1 já esteja disponível
            if (
                st.session_state.df_atividade_2 is None
                and st.session_state.df_converted is not None
            ):
                # Botão para carregar o DataFrame da Atividade 1
                if st.button(
                    "📥 Carregar DataFrame da Atividade 1", key="load_df_button"
                ):
                    st.session_state.df_atividade_2 = st.session_state.df_converted
                    st.success("✅ DataFrame carregado com sucesso!")
                    st.rerun()  # Recarrega a aplicação para refletir as mudanças

            # Caso o DataFrame da Atividade 2 já esteja disponível
            if st.session_state.df_atividade_2 is not None:
                df_atividade_2 = st.session_state.df_atividade_2
                coluna_1, coluna_2 = st.columns(2)

                # Primeira coluna -> visualização do DataFrame final
                with coluna_1:
                    st.write("#### **Visual do dataframe final**")
                    st.dataframe(st.session_state.df_atividade_2)

                # Segunda coluna -> Exibição dos metadados (tipos, colunas, missings)
                with coluna_2:
                    st.write("#### **Metadados definitivo**")
                    df_atividade_2 = st.session_state.df_atividade_2
                    metadados_definitivo = {
                        "Colunas": df_atividade_2.columns,
                        "Tipo dos dados": df_atividade_2.dtypes.values,
                        "n_missings": df_atividade_2.isna().sum().values,
                    }
                    metadados_definitivo_df = pd.DataFrame(metadados_definitivo)
                    st.dataframe(metadados_definitivo_df)

                    # Verificação de valores ausentes (missing values)
                    if metadados_definitivo_df["n_missings"].sum() == 0:
                        st.session_state.show_success = True
                        if st.session_state.show_success:
                            success_container.success("Não há valores missings")
                            time.sleep(0.5)  # Mantém a mensagem por 2 segundos
                            success_container.empty()
                            st.session_state.show_success = False
                    else:
                        st.warning(
                            f"⚠️ {df_atividade_2.isna().sum().sum()} valores missing"
                        )

            # Nova validação caso o DataFrame esteja carregado
            if st.session_state.df_atividade_2 is not None:
                df_atividade_2 = st.session_state.df_atividade_2

                # Verifica se existem colunas categóricas que precisam ser convertidas em dummies
                check_possiveis_dummies = [
                    coluna
                    for coluna in df_atividade_2.select_dtypes(
                        exclude=[np.number]
                    ).columns
                    if df_atividade_2[coluna].nunique() > 2  # Mais de 2 categorias
                ]

                # Caso existam colunas categóricas problemáticas
                if check_possiveis_dummies:
                    st.warning(
                        f"Temos presente {len(check_possiveis_dummies)} que não se adequam aos padrões do scikit-learn por serem strings"
                    )
                else:
                    st.session_state.show_success = True
                    if st.session_state.show_success:
                        success_container.success(
                            "Não há strings dentre as variáveis do dataframe"
                        )
                        time.sleep(0.5)
                        success_container.empty()
                        st.session_state.show_success = False

                # Se não houver problemas de missings e strings, permite a divisão treino/teste
                if (len(check_possiveis_dummies) == 0) and (
                    df_atividade_2.isna().sum().sum() == 0
                ):
                    if st.button("Dividir o dataframe entre teste e treino"):
                        df_atividade_2 = st.session_state.df_atividade_2.copy()

                        # Define X (features) e y (target)
                        x = df_atividade_2.drop(columns=["mau"])
                        y = df_atividade_2["mau"]

                        # Faz a divisão entre treino (70%) e teste (30%)
                        x_train, x_test, y_train, y_test = train_test_split(
                            x, y, test_size=0.3, random_state=100
                        )

                        st.session_state.x_train = x_train
                        st.session_state.y_train = y_train
                        st.session_state.x_test = x_test
                        st.session_state.y_test = y_test

                        # Exibe mensagem de sucesso
                        st.session_state.show_success = True
                        if st.session_state.show_success:
                            success_container.success(
                                "Foram criadas as divisões de teste e treino da base de dados"
                            )
                            st.session_state.verificador_string = True
                            time.sleep(0.5)
                            success_container.empty()
                            st.session_state.show_success = False

                        # Mostra resumo da divisão
                        st.divider()
                        st.write("#### Dados da divisão do dataframe")
                        st.session_state.data_ready = True
                        coluna_treino, coluna_teste = st.columns(2)

                        with coluna_treino:
                            st.write("**Conjunto de dados para treino**")
                        with coluna_teste:
                            st.write("**Conjunto de dados para teste**")

                        # Métricas em colunas para treino e teste
                        (
                            coluna_treino_1,
                            coluna_treino_2,
                            coluna_treino_3,
                            coluna_teste_1,
                            coluna_teste_2,
                            coluna_teste_3,
                        ) = st.columns(6)

                        # Informações do conjunto de treino
                        with coluna_treino_1:
                            st.metric("Número de linhas:", x_train.shape[0])
                        with coluna_treino_2:
                            st.metric("Número de colunas:", x_train.shape[1])
                        with coluna_treino_3:
                            porcentagem_df_treino = round(
                                (x_train.shape[0] / df_atividade_2.shape[0]) * 100, 2
                            )
                            st.metric(
                                "Porcentgem do dataframe original",
                                porcentagem_df_treino,
                            )

                        # Informações do conjunto de teste
                        with coluna_teste_1:
                            st.metric("Número de linhas:", x_test.shape[0])
                        with coluna_teste_2:
                            st.metric("Número de colunas:", x_test.shape[1])
                        with coluna_teste_3:
                            porcentagem_df_teste = round(
                                (x_test.shape[0] / df_atividade_2.shape[0]) * 100, 2
                            )
                            st.metric(
                                "Porcentgem do dataframe original", porcentagem_df_teste
                            )
        with aba_2_construcao_e_visualizacao_do_modelo:
            st.markdown("#### Construção e exibição do modelo")
            selecao_de_arvore = st.radio(
                "Selecione o tipo de árvore para avaliarmos",
                ["Árvore sem poda", "Árvore com poda"],
                horizontal=True,
            )
            if (st.session_state.df_atividade_2 is not None) and (
                st.session_state.data_ready is not None
            ):
                try:

                    if selecao_de_arvore == "Árvore sem poda":
                        clf = None
                        clf = DecisionTreeClassifier(random_state=100)
                        clf.fit(st.session_state.x_train, st.session_state.y_train)
                    elif selecao_de_arvore == "Árvore com poda":
                        clf = DecisionTreeClassifier(
                            max_depth=10, min_samples_leaf=5, random_state=123
                        )
                        clf.fit(st.session_state.x_train, st.session_state.y_train)
                    st.session_state.show_success = True
                    if st.session_state.show_success:
                        success_container.success(
                            "Sucesso na criação e fittagem do modelo"
                        )
                        time.sleep(0.5)
                        st.session_state.show_success = False
                except Exception as e:
                    st.error("❌ Não foi possível criar o modelo")
                    st.exception(e)

                if clf is not None and hasattr(clf, "tree_"):
                    fig, ax = plt.subplots(figsize=(35, 20))
                    plot_tree(
                        clf,
                        filled=True,
                        class_names=["aprovados", "reprovados"],
                        feature_names=list(st.session_state.x_train.columns),
                    )

                    buf_tree = BytesIO()
                    fig.savefig(buf_tree, format="png", bbox_inches="tight")
                    buf_tree.seek(0)

                    predict_train = clf.predict(st.session_state.x_train)
                    predict_test = clf.predict(st.session_state.x_test)

                    with st.expander("Visualizar os dados do modelo", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Árvore gerada**")
                            st.image(buf_tree, width=800)

                        with col2:

                            selecao_matriz = st.radio(
                                "Selecione qual matriz da confusão será exíbida",
                                [
                                    "Matriz do conjunto de treino",
                                    "Matriz do conjunto teste",
                                ],
                                horizontal=True,
                            )
                            if selecao_matriz == "Matriz do conjunto de treino":
                                cm = confusion_matrix(
                                    st.session_state.y_train,
                                    predict_train,
                                    labels=clf.classes_,
                                )
                                MatrizDeConfusao = ConfusionMatrixDisplay(
                                    confusion_matrix=cm
                                )
                            if selecao_matriz == "Matriz do conjunto teste":
                                cm = confusion_matrix(
                                    st.session_state.y_test,
                                    predict_test,
                                    labels=clf.classes_,
                                )
                                MatrizDeConfusao = ConfusionMatrixDisplay(
                                    confusion_matrix=cm
                                )
                            fig, ax = plt.subplots()
                            MatrizDeConfusao.plot(ax=ax)
                            buf_confusion_matrix_train = BytesIO()
                            fig.savefig(
                                buf_confusion_matrix_train,
                                format="png",
                                bbox_inches="tight",
                            )
                            buf_confusion_matrix_train.seek(0)
                            st.write("**Matriz de confusão treino**")
                            st.image(buf_confusion_matrix_train, width=800)
                        st.divider()
                        st.write("#### Performance do modelo:")
                        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
                        with col1:
                            st.metric(
                                "Acurácia do modelo na base de treino é de:",
                                round(
                                    accuracy_score(
                                        predict_train, st.session_state.y_train
                                    ),
                                    3,
                                ),
                            )
                        with col2:
                            st.metric(
                                "Acurácia do modelo na base de testes é de:",
                                round(
                                    accuracy_score(
                                        predict_test, st.session_state.y_test
                                    ),
                                    3,
                                ),
                            )
                        with col3:
                            st.metric(
                                "A diferença entre as duas performances:",
                                round(
                                    accuracy_score(
                                        predict_train, st.session_state.y_train
                                    )
                                    - accuracy_score(
                                        predict_test, st.session_state.y_test
                                    ),
                                    3,
                                ),
                            )
                        with col4:
                            st.write(
                                "Dataframe representando a porcentagem de aprovados e reprovados do modelo"
                            )
                            cm_df = pd.DataFrame(
                                cm,
                                index=["Reais: Aprovados", "Reais: Reprovados"],
                                columns=[
                                    "Previstos: Aprovados",
                                    "Previstos: Reprovados",
                                ],
                            )

                            cm_df_pct = cm_df.astype("float") / cm.sum() * 100
                            st.dataframe(cm_df_pct)

            elif st.session_state.df_atividade_2 is None:
                st.warning("Os dados não foram carregados")
            elif (st.session_state.df_atividade_2 is not None) and (
                st.session_state.data_ready is None
            ):
                st.warning("Os dados não estão adequados para a construção do modelo")


if __name__ == "__main__":
    main()
