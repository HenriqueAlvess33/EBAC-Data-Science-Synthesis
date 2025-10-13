import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy.stats import mode
import io

# Configuração da página
st.set_page_config(page_title="Classificador com Árvores de Decisão", layout="wide")

# Título da aplicação
st.title("🌳 Classificador Ensemble com Árvores de Decisão")
st.markdown("---")

# Sidebar para upload e configurações
st.sidebar.header("📁 Upload de Dados")

# Opção: usar dados de exemplo ou upload
opcao_dados = st.sidebar.radio(
    "Escolha a fonte dos dados:", ["Usar Dados de Exemplo", "Fazer Upload de Arquivo"]
)

df = None
teste = None

if opcao_dados == "Fazer Upload de Arquivo":
    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do arquivo CSV para treinamento",
        type=["csv"],
        help="Arquivo CSV com as features e coluna alvo",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Arquivo carregado com sucesso!")

            # Upload do arquivo de teste
            uploaded_file_teste = st.sidebar.file_uploader(
                "Faça upload do arquivo CSV para teste (opcional)",
                type=["csv"],
                help="Arquivo CSV com dados para previsão",
            )

            if uploaded_file_teste is not None:
                teste = pd.read_csv(uploaded_file_teste)
                st.sidebar.success("Arquivo de teste carregado!")

        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {e}")

else:
    # Usar dados de exemplo
    st.sidebar.info("Usando dados de exemplo aleatórios")

    # Criar DataFrame de exemplo
    df = pd.DataFrame(
        np.random.randn(9, 4) * 100,
        columns=["coluna1", "coluna2", "coluna3", "coluna4"],
    )

    # Criar conjunto de teste
    teste = pd.DataFrame(
        np.random.randn(9, 4) * 100,
        columns=["coluna1", "coluna2", "coluna3", "coluna4"],
    )

# Configurações do modelo
st.sidebar.header("⚙️ Configurações do Modelo")
n_amostras = st.sidebar.slider("Número de amostras por árvore:", 5, 20, 9)
max_profundidade = st.sidebar.slider("Profundidade máxima das árvores:", 1, 5, 2)
n_arvores = st.sidebar.slider("Número de árvores no ensemble:", 2, 10, 4)

if df is not None:
    # Adicionar coluna alvo se não existir
    if "Interesse" not in df.columns:
        st.sidebar.warning(
            "Coluna 'Interesse' não encontrada. Gerando valores aleatórios..."
        )
        df["Interesse"] = np.random.choice([True, False], size=(len(df),))

    # Mostrar dados
    st.header("📊 Dados de Treinamento")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visualização dos Dados")
        st.dataframe(df, use_container_width=True)

    with col2:
        st.subheader("Estatísticas Descritivas")
        st.dataframe(df.describe(), use_container_width=True)

    # Mostrar dados de teste se disponíveis
    if teste is not None:
        st.header("🧪 Dados de Teste")
        st.dataframe(teste, use_container_width=True)

    # Processamento do modelo
    st.header("🔧 Treinamento do Modelo")

    # Gerar amostras bootstrap
    st.subheader(f"Gerando {n_arvores} amostras bootstrap...")

    amostras = []
    x_amostras = []
    y_amostras = []
    modelos = []

    # Criar colunas para organizar as amostras
    cols_amostras = st.columns(min(n_arvores, 4))

    for i in range(n_arvores):
        amostra = df.sample(n=n_amostras, replace=True, random_state=100 + i)
        amostras.append(amostra)

        x_amostra = amostra[["coluna1", "coluna2", "coluna3", "coluna4"]]
        y_amostra = amostra["Interesse"]

        x_amostras.append(x_amostra)
        y_amostras.append(y_amostra)

        # Mostrar amostra em coluna
        if i < len(cols_amostras):
            with cols_amostras[i]:
                st.metric(f"Amostra {i+1}", f"{len(amostra)} registros")
                st.dataframe(
                    amostra[["coluna1", "coluna2", "Interesse"]].head(3),
                    use_container_width=True,
                )

    # Treinar modelos
    st.subheader(f"Treinando {n_arvores} árvores de decisão...")

    progress_bar = st.progress(0)

    for i in range(n_arvores):
        clf = DecisionTreeClassifier(random_state=100 + i, max_depth=max_profundidade)
        modelo = clf.fit(x_amostras[i], y_amostras[i])
        modelos.append(modelo)
        progress_bar.progress((i + 1) / n_arvores)

    st.success("Modelos treinados com sucesso!")

    # Visualização das árvores
    st.header("🌲 Visualização das Árvores de Decisão")

    # Criar abas para cada árvore
    tabs = st.tabs([f"Árvore {i+1}" for i in range(n_arvores)])

    for i, tab in enumerate(tabs):
        with tab:
            st.subheader(f"Árvore de Decisão - Modelo {i+1}")

            # Criar figura
            fig, ax = plt.subplots(figsize=(20, 8))
            plot_tree(
                modelos[i],
                filled=True,
                class_names=["False", "True"],
                feature_names=["coluna1", "coluna2", "coluna3", "coluna4"],
                ax=ax,
            )
            plt.title(f"Árvore {i+1} - Profundidade: {max_profundidade}")
            st.pyplot(fig)

    # Previsões e ensemble
    if teste is not None:
        st.header("🎯 Previsões do Ensemble")

        # Fazer previsões
        st.subheader("Previsões Individuais")

        predicoes = []
        cols_pred = st.columns(n_arvores)

        for i in range(n_arvores):
            pred = modelos[i].predict(teste)
            pred_int = pred.astype(int)
            predicoes.append(pred_int)

            with cols_pred[i]:
                st.metric(f"Modelo {i+1}", f"{np.sum(pred)} True")
                st.dataframe(
                    pd.DataFrame({"Amostra": range(len(pred)), "Previsão": pred}),
                    use_container_width=True,
                    height=200,
                )

        # Votação por maioria
        st.subheader("Votação por Maioria (Ensemble)")

        todas_predicoes = np.vstack(predicoes)
        predicao_final, contagem = mode(todas_predicoes, axis=0, keepdims=False)

        # Converter para boolean
        predicao_final_bool = predicao_final.astype(bool)

        # Mostrar resultados finais
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.metric("Previsões True", f"{np.sum(predicao_final_bool)}")
            st.metric(
                "Previsões False",
                f"{len(predicao_final_bool) - np.sum(predicao_final_bool)}",
            )

        with col_res2:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Amostra": range(len(predicao_final_bool)),
                        "Previsão Final": predicao_final_bool,
                        "Votos True": np.sum(todas_predicoes, axis=0),
                        "Votos False": n_arvores - np.sum(todas_predicoes, axis=0),
                    }
                ),
                use_container_width=True,
            )

        # Matriz de concordância
        st.subheader("Matriz de Concordância entre Modelos")
        concordancia = pd.DataFrame(
            todas_predicoes.T, columns=[f"Modelo {i+1}" for i in range(n_arvores)]
        )
        st.dataframe(concordancia, use_container_width=True)

    # Informações do ensemble
    st.sidebar.header("📈 Métricas do Ensemble")
    st.sidebar.metric("Número de Árvores", n_arvores)
    st.sidebar.metric("Tamanho das Amostras", n_amostras)
    st.sidebar.metric("Profundidade Máxima", max_profundidade)

    # Explicação do método
    with st.expander("ℹ️ Sobre o Método Ensemble"):
        st.markdown(
            """
        ### Bagging (Bootstrap Aggregating)
        
        Este método cria múltiplos modelos usando amostras diferentes dos dados de treinamento:
        
        1. **Amostragem Bootstrap**: Cada árvore é treinada com uma amostra aleatória com reposição
        2. **Árvores Independentes**: Cada árvore é treinada independentemente
        3. **Votação por Maioria**: A previsão final é determinada pela maioria dos votos
        
        **Vantagens:**
        - Reduz overfitting
        - Aumenta estabilidade do modelo
        - Melhora performance em dados não vistos
        """
        )

else:
    # Instruções iniciais
    st.info("👆 Configure os dados e parâmetros na sidebar para começar")

    # Exemplo de estrutura de dados esperada
    st.header("📋 Estrutura Esperada dos Dados")

    exemplo_data = {
        "coluna1": [120.5, -80.3, 45.7, -23.1, 167.8],
        "coluna2": [-45.2, 89.1, -12.4, 67.8, -34.5],
        "coluna3": [78.9, -56.7, 23.4, -91.2, 45.6],
        "coluna4": [-12.3, 34.5, -67.8, 89.0, -23.4],
        "Interesse": [True, False, True, False, True],
    }
    exemplo_df = pd.DataFrame(exemplo_data)
    st.dataframe(exemplo_df, use_container_width=True)

    st.markdown(
        """
    **Requisitos do arquivo CSV:**
    - Colunas de features: `coluna1`, `coluna2`, `coluna3`, `coluna4` (ou nomes similares)
    - Coluna alvo: `Interesse` (valores booleanos True/False)
    - Dados numéricos para as features
    """
    )
