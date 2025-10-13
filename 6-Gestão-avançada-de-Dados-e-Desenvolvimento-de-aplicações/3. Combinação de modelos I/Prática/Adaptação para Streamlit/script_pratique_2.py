import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from scipy.stats import mode

# Configuração da página
st.set_page_config(page_title="Classificador com Árvores de Decisão", layout="wide")
st.title("📊 Classificador com Árvores de Decisão - Ensemble")

# Sidebar para upload de arquivos
st.sidebar.header("📁 Upload de Dados")
uploaded_file = st.sidebar.file_uploader(
    "Faça upload do seu arquivo CSV",
    type=["csv"],
    help="Arquivo CSV com os dados para treinamento",
)

# Parâmetros do modelo
st.sidebar.header("⚙️ Parâmetros do Modelo")
max_depth = st.sidebar.slider("Profundidade máxima das árvores", 1, 5, 2)
n_models = st.sidebar.slider("Número de modelos", 2, 6, 4)
target_column = st.sidebar.text_input("Nome da coluna target", "Interesse")


def criar_dataframe_padrao():
    """Cria um DataFrame padrão caso nenhum arquivo seja carregado"""
    df = pd.DataFrame(
        np.random.randn(9, 4) * 100,
        columns=["coluna1", "coluna2", "coluna3", "coluna4"],
    )
    df["Interesse"] = np.random.choice([True, False], size=(9,))
    return df


def processar_dados(uploaded_file):
    """Processa os dados do arquivo upload ou cria dados padrão"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado com sucesso! Shape: {df.shape}")
            return df
        except Exception as e:
            st.error(f"❌ Erro ao carregar arquivo: {e}")
            st.info("📝 Usando dados padrão...")
            return criar_dataframe_padrao()
    else:
        st.info("📝 Nenhum arquivo carregado. Usando dados padrão...")
        return criar_dataframe_padrao()


def criar_amostragens(df, n_models, target_column):
    """Cria amostragens e conjuntos de teste"""
    amostragens = []
    testes = []

    for i in range(n_models):
        # Amostragem com reposição
        amostra = df.sample(n=len(df), replace=True)

        # Seleciona 3 colunas aleatórias (excluindo a target se presente)
        colunas_disponiveis = [col for col in amostra.columns if col != target_column]
        colunas_selecionadas = np.random.choice(
            colunas_disponiveis, size=min(3, len(colunas_disponiveis)), replace=False
        )

        amostra_final = amostra[colunas_selecionadas]
        if target_column in amostra.columns:
            amostra_final[target_column] = amostra[target_column]

        amostragens.append(amostra_final)

        # Cria conjunto de teste correspondente
        teste = pd.DataFrame(
            np.random.randn(len(df), len(colunas_selecionadas)) * 100,
            columns=colunas_selecionadas,
        )
        testes.append(teste)

    return amostragens, testes


def treinar_modelos(amostragens, target_column, max_depth):
    """Treina os modelos de árvore de decisão"""
    modelos = []
    X_sets = []
    y_sets = []

    for i, amostra in enumerate(amostragens):
        # Separa features e target
        if target_column in amostra.columns:
            X = amostra.drop(columns=[target_column])
            y = amostra[target_column]
        else:
            X = amostra
            # Usa target do DataFrame original ou cria um padrão
            y = pd.Series(np.random.choice([True, False], size=len(amostra)))

        X_sets.append(X)
        y_sets.append(y)

        # Treina o modelo
        clf = DecisionTreeClassifier(random_state=100 + i, max_depth=max_depth)
        clf.fit(X, y)
        modelos.append(clf)

    return modelos, X_sets, y_sets


def plotar_arvores(modelos, X_sets):
    """Plota as árvores de decisão"""
    for i, (modelo, X) in enumerate(zip(modelos, X_sets)):
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(
            modelo,
            filled=True,
            class_names=["False", "True"],
            feature_names=X.columns,
            ax=ax,
        )
        ax.set_title(f"Árvore de Decisão - Modelo {i+1}", fontsize=16, pad=20)
        st.pyplot(fig)


def fazer_previsoes(modelos, testes):
    """Faz previsões e aplica votação por maioria"""
    todas_predicoes = []

    for i, (modelo, teste) in enumerate(zip(modelos, testes)):
        pred = modelo.predict(teste)
        pred_int = pred.astype(int)
        todas_predicoes.append(pred_int)

        # Mostra previsões individuais
        st.write(f"**Previsões do Modelo {i+1}:**")
        st.write(
            pd.DataFrame(
                {
                    "Amostra": range(1, len(pred) + 1),
                    "Previsão": pred,
                    "Previsão (Inteiro)": pred_int,
                }
            )
        )

    # Votação por maioria
    todas_predicoes_stack = np.vstack(todas_predicoes)
    predicao_final, _ = mode(todas_predicoes_stack, axis=0, keepdims=False)

    return predicao_final, todas_predicoes_stack


# Processamento principal
df = processar_dados(uploaded_file)

# Mostrar dados
st.header("📋 Dados Utilizados")
st.dataframe(df, use_container_width=True)

st.write(f"**Shape do DataFrame:** {df.shape}")
st.write(f"**Colunas disponíveis:** {list(df.columns)}")

# Verificar se a coluna target existe
if target_column not in df.columns:
    st.warning(
        f"⚠️ Coluna '{target_column}' não encontrada no DataFrame. Será criada automaticamente."
    )
    df[target_column] = np.random.choice([True, False], size=len(df))

# Criar amostragens e treinar modelos
if st.button("🎯 Treinar Modelos e Fazer Previsões"):
    with st.spinner("Processando..."):
        # Criar amostragens
        amostragens, testes = criar_amostragens(df, n_models, target_column)

        # Treinar modelos
        modelos, X_sets, y_sets = treinar_modelos(amostragens, target_column, max_depth)

        # Mostrar informações das amostragens
        st.header("📊 Informações das Amostragens")
        for i, amostra in enumerate(amostragens):
            st.write(f"**Amostragem {i+1}:** {amostra.shape[1]} features")
            st.write(f"Features: {list(amostra.columns)}")

        # Plotar árvores
        st.header("🌳 Visualização das Árvores de Decisão")
        plotar_arvores(modelos, X_sets)

        # Fazer previsões
        st.header("🔮 Previsões")
        predicao_final, todas_predicoes = fazer_previsoes(modelos, testes)

        # Mostrar resultado final
        st.header("🏆 Resultado Final - Votação por Maioria")
        resultado_final = pd.DataFrame(
            {
                "Amostra": range(1, len(predicao_final) + 1),
                "Previsão Final": predicao_final,
                "Previsão Final (Bool)": predicao_final.astype(bool),
            }
        )
        st.dataframe(resultado_final, use_container_width=True)

        # Estatísticas
        st.subheader("📈 Estatísticas das Previsões")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total de Amostras", len(predicao_final))
        with col2:
            verdadeiros = np.sum(predicao_final)
            st.metric("Previsões 'True'", verdadeiros)
        with col3:
            falsos = len(predicao_final) - verdadeiros
            st.metric("Previsões 'False'", falsos)

# Informações adicionais
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Sobre a Aplicação")
st.sidebar.info(
    """
Esta aplicação implementa um ensemble de árvores de decisão com:
- **Bagging**: Amostragens com reposição
- **Votação por maioria**: Combina previsões dos modelos
- **Upload de dados**: Use seus próprios dados CSV
"""
)
