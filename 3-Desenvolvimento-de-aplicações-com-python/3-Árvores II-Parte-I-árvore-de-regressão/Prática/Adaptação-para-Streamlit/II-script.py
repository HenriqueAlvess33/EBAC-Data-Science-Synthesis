import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree

# ------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ------------------------------------------------------
st.set_page_config(
    page_title="Modelando com scikit-learn",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="varig_icon.png",
)


# ------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------
def load_data(uploaded_file):
    """Melhoria: Adiciona mais informações sobre o formato esperado"""
    try:
        data = pd.read_csv(uploaded_file)
        st.success(f"Dados carregados com sucesso! Shape: {data.shape}")
        return data
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None


def preprocess_data(data):
    """Separa o pré-processamento em função própria"""
    data_clean = data.dropna()

    # Verifica se a coluna target existe
    if "median_house_value" not in data_clean.columns:
        st.error("Coluna 'median_house_value' não encontrada nos dados!")
        return None, None

    x = data_clean.drop(columns=["median_house_value"])
    x = pd.get_dummies(x, drop_first=True)
    y = data_clean["median_house_value"]

    return x, y


def plot_performance(ccp_alphas, train_scores, test_scores):
    """Melhoria: Função dedicada para plotar performance"""
    fig, ax = plt.subplots(figsize=[12, 6])

    ax.set_xlabel("Alpha", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("MSE x Alpha - Treino vs Teste", fontsize=14)

    ax.plot(ccp_alphas[:-1], train_scores[:-1], marker="o", label="treino", linewidth=2)
    ax.plot(ccp_alphas[:-1], test_scores[:-1], marker="o", label="teste", linewidth=2)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=12)

    # Melhoria: Formatação dos eixos
    ax.tick_params(axis="both", which="major", labelsize=10)

    return fig


# ------------------------------------------------------
# FUNÇÃO PRINCIPAL DO APP
# ------------------------------------------------------
def main():
    # Cabeçalho
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🌳 Árvores de Regressão com scikit-learn")
        st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Configurações")
        uploaded_file = st.file_uploader(
            "Upload do arquivo CSV",
            type="csv",
            help="Faça upload do arquivo housing.csv do repositório",
        )

        # Melhoria: Adicionar configurações do modelo
        st.subheader("Parâmetros do Modelo")
        test_size = st.slider("Tamanho do teste:", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state:", 0, 1000, 100)

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        if data is not None:
            # Melhoria: Mostrar informações dos dados
            with st.expander("📊 Visualização dos Dados", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Linhas", data.shape[0])
                with col2:
                    st.metric("Colunas", data.shape[1])
                with col3:
                    st.metric("Valores Faltantes", data.isna().sum().sum())

                st.subheader("Primeiras linhas:")
                st.dataframe(data.head())

            # Pré-processamento
            x, y = preprocess_data(data)

            if x is not None and y is not None:
                # Divisão dos dados
                X_, X_test, y_, y_test = train_test_split(
                    x, y, test_size=test_size, random_state=random_state
                )
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X_, y_, test_size=0.2, random_state=random_state
                )

                # Modelos baseline
                with st.expander("🎯 Modelos Basilares", expanded=True):
                    st.write("### Comparação de Modelos Iniciais")

                    regr_1 = DecisionTreeRegressor(
                        max_depth=8, random_state=random_state
                    )
                    regr_2 = DecisionTreeRegressor(
                        max_depth=2, random_state=random_state
                    )

                    regr_1.fit(X_train, y_train)
                    regr_2.fit(X_train, y_train)

                    # Cálculo de métricas
                    r2_1_train = regr_1.score(X_train, y_train)
                    r2_2_train = regr_2.score(X_train, y_train)
                    r2_1_test = regr_1.score(X_test, y_test)
                    r2_2_test = regr_2.score(X_test, y_test)

                    # Melhoria: Layout mais informativo
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📈 Dados de Treino")
                        st.metric(
                            "Árvore (profundidade=8)",
                            f"{r2_1_train:.1%}",
                            help="R² score nos dados de treino",
                        )
                        st.metric(
                            "Árvore (profundidade=2)",
                            f"{r2_2_train:.1%}",
                            help="R² score nos dados de treino",
                        )

                    with col2:
                        st.subheader("🧪 Dados de Teste")
                        st.metric(
                            "Árvore (profundidade=8)",
                            f"{r2_1_test:.1%}",
                            delta=f"{(r2_1_test - r2_1_train):.3f}",
                            delta_color="inverse",
                        )
                        st.metric(
                            "Árvore (profundidade=2)",
                            f"{r2_2_test:.1%}",
                            delta=f"{(r2_2_test - r2_2_train):.3f}",
                            delta_color="inverse",
                        )

                # Podagem de árvores
                st.markdown("---")
                with st.expander("🌿 Otimização por Podagem", expanded=True):
                    st.write("### Seleção do Melhor Alpha por Podagem")

                    path = regr_1.cost_complexity_pruning_path(X_train, y_train)
                    ccp_alphas, impurities = path.ccp_alphas, path.impurities

                    @st.cache_data
                    def train_multiple_trees(ccp_alphas, X_train, y_train):
                        clfs = []
                        for ccp_alpha in ccp_alphas:
                            clf = DecisionTreeRegressor(
                                random_state=random_state, ccp_alpha=ccp_alpha
                            )
                            clf.fit(X_train, y_train)
                            clfs.append(clf)
                        return clfs

                    clfs = train_multiple_trees(
                        ccp_alphas=ccp_alphas, X_train=X_train, y_train=y_train
                    )

                    train_scores = [
                        mean_squared_error(y_train, clf.predict(X_train))
                        for clf in clfs
                    ]
                    test_scores = [
                        mean_squared_error(y_test, clf.predict(X_test)) for clf in clfs
                    ]

                    # Plot de performance
                    fig = plot_performance(ccp_alphas, train_scores, test_scores)
                    st.pyplot(fig)

                    # Melhoria: Seleção automática do melhor alpha
                    best_alpha_idx = np.argmin(test_scores[:-1])
                    best_alpha = ccp_alphas[best_alpha_idx]

                    st.info(f"🎯 Melhor alpha encontrado: {best_alpha:.4f}")

                # Modelo final
                with st.expander("🏆 Modelo Final", expanded=True):
                    st.write("### Performance do Modelo Otimizado")

                    # Usa o melhor alpha encontrado
                    arvore_final = DecisionTreeRegressor(
                        random_state=random_state,
                        ccp_alpha=best_alpha,  # Usa o melhor alpha automaticamente
                    )
                    arvore_final.fit(X_train, y_train)

                    r2_final_train = arvore_final.score(X_train, y_train)
                    r2_final_test = arvore_final.score(X_test, y_test)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Treino", f"{r2_final_train:.1%}")
                    with col2:
                        st.metric("R² Teste", f"{r2_final_test:.1%}")
                    with col3:
                        overfitting = r2_final_train - r2_final_test
                        st.metric(
                            "Overfitting",
                            f"{overfitting:.3f}",
                            delta_color="inverse" if overfitting > 0.1 else "normal",
                        )

                    # Visualização da árvore
                    st.write("### Diagrama da Árvore Final")
                    fig_tree, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(
                        arvore_final,
                        feature_names=x.columns,
                        filled=True,
                        ax=ax,
                        fontsize=8,
                        proportion=True,
                    )
                    plt.tight_layout()

                    buff = BytesIO()
                    plt.savefig(buff, format="png", dpi=150, bbox_inches="tight")
                    buff.seek(0)
                    st.image(buff)
                    plt.close(fig_tree)

                    # Melhoria: Botão para download da imagem
                    st.download_button(
                        label="📥 Download do Diagrama",
                        data=buff,
                        file_name="arvore_final.png",
                        mime="image/png",
                    )


if __name__ == "__main__":
    main()
