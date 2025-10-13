import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_gaussian_quantiles, make_regression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

# Configuração da página
st.set_page_config(page_title="AdaBoost Analysis", layout="wide")
st.title("Análise do Algoritmo AdaBoost")

# Sidebar para configurações
st.sidebar.header("Configurações do Modelo")
dataset_choice = st.sidebar.selectbox("Escolha o dataset:", ["Iris", "Sintético Gaussian"])

# Função para calcular erro de classificação
def misclassification_error(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)

# Abas para organizar as visualizações
tab1, tab2, tab3, tab4 = st.tabs([
    "Classificação Básica", 
    "Convergência do AdaBoost", 
    "Weak Learners", 
    "Otimização de Hiperparâmetros"
])

with tab1:
    st.header("Classificação com AdaBoost")
    
    if dataset_choice == "Iris":
        X, y = load_iris(return_X_y=True)
        dataset_name = "Iris"
    else:
        X, y = make_gaussian_quantiles(n_samples=2000, n_features=10, n_classes=3, random_state=1)
        dataset_name = "Sintético Gaussian"
    
    st.write(f"Dataset: {dataset_name}")
    st.write(f"Shape dos dados: {X.shape}")
    
    # Validação cruzada
    clf = AdaBoostClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Acurácia Média (Validação Cruzada)", f"{scores.mean():.3f}")
        st.metric("Desvio Padrão", f"{scores.std():.3f}")
    
    with col2:
        st.write("Acurácias por fold:")
        scores_df = pd.DataFrame({
            'Fold': range(1, 6),
            'Acurácia': scores
        })
        st.dataframe(scores_df, hide_index=True)

with tab2:
    st.header("Análise de Convergência do AdaBoost")
    
    # Gerar dados sintéticos para análise de convergência
    X, y = make_gaussian_quantiles(n_samples=2000, n_features=10, n_classes=3, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=420)
    
    # Configurações do modelo
    max_depth = st.slider("Profundidade máxima do Weak Learner", 1, 5, 1)
    max_leaf_nodes = st.slider("Número máximo de folhas", 2, 20, 8)
    n_estimators = st.slider("Número de estimadores", 10, 200, 100)
    
    weak_learner = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    
    # Treinar modelos
    adaboost_clf = AdaBoostClassifier(
        estimator=weak_learner, n_estimators=n_estimators, random_state=42
    ).fit(X_train, y_train)
    
    dummy_clf = DummyClassifier().fit(X_train, y_train)
    
    # Calcular erros
    weak_learner_error = misclassification_error(
        y_test, weak_learner.fit(X_train, y_train).predict(X_test)
    )
    
    dummy_error = misclassification_error(
        y_test, dummy_clf.predict(X_test)
    )
    
    # DataFrame para erros do AdaBoost
    boosting_errors = pd.DataFrame({
        "Number of trees": range(1, n_estimators + 1),
        "AdaBoost": [
            misclassification_error(y_test, y_pred)
            for y_pred in adaboost_clf.staged_predict(X_test)
        ],
    }).set_index("Number of trees")
    
    # Plotar gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    boosting_errors.plot(ax=ax)
    ax.set_ylabel("Erro de Classificação no Conjunto de Teste")
    ax.set_xlabel("Número de Árvores")
    ax.set_title("Convergência do Algoritmo AdaBoost")
    
    # Linhas de referência
    plt.plot(
        [boosting_errors.index.min(), boosting_errors.index.max()],
        [weak_learner_error, weak_learner_error],
        "k--",
        label="Weak learner",
        color="tab:orange",
    )
    
    plt.plot(
        [boosting_errors.index.min(), boosting_errors.index.max()],
        [dummy_error, dummy_error],
        "k:",
        label="Dummy classifier",
        color="c",
    )
    
    plt.legend(["AdaBoost", "DecisionTreeClassifier", "DummyClassifier"], loc=1)
    plt.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Erro do Weak Learner", f"{weak_learner_error:.3f}")
    with col2:
        st.metric("Erro do Dummy Classifier", f"{dummy_error:.3f}")
    with col3:
        final_ada_error = misclassification_error(y_test, adaboost_clf.predict(X_test))
        st.metric("Erro Final do AdaBoost", f"{final_ada_error:.3f}")

with tab3:
    st.header("Análise dos Weak Learners")
    
    # Usar o modelo já treinado da aba anterior
    if 'adaboost_clf' in locals():
        weak_learners_info = pd.DataFrame({
            "Number of trees": range(1, n_estimators + 1),
            "Errors": adaboost_clf.estimator_errors_,
            "Weights": adaboost_clf.estimator_weights_,
        }).set_index("Number of trees")
        
        # Plotar gráficos dos weak learners
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        weak_learners_info["Errors"].plot(ax=ax1, color="tab:blue")
        ax1.set_ylabel("Erro de Treino")
        ax1.set_xlabel("Número de Árvores")
        ax1.set_title("Erro dos Weak Learners")
        ax1.grid(True, alpha=0.3)
        
        weak_learners_info["Weights"].plot(ax=ax2, color="tab:blue")
        ax2.set_ylabel("Peso")
        ax2.set_xlabel("Número de Árvores")
        ax2.set_title("Pesos dos Weak Learners")
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle("Erros e Pesos dos Weak Learners no AdaBoost")
        fig.tight_layout()
        
        st.pyplot(fig)
        
        # Estatísticas
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Estatísticas dos Erros:**")
            st.write(weak_learners_info["Errors"].describe())
        with col2:
            st.write("**Estatísticas dos Pesos:**")
            st.write(weak_learners_info["Weights"].describe())
    else:
        st.info("Treine um modelo na aba 'Convergência do AdaBoost' primeiro.")

with tab4:
    st.header("Otimização de Hiperparâmetros")
    
    X, y = load_iris(return_X_y=True)
    
    st.write("Busca em Grade para encontrar os melhores hiperparâmetros:")
    
    params = {
        "n_estimators": [50, 100, 200], 
        "learning_rate": [0.01, 0.1, 1]
    }
    
    if st.button("Executar Grid Search"):
        with st.spinner("Executando busca em grade..."):
            grid_adaboost = GridSearchCV(
                estimator=AdaBoostClassifier(random_state=42), 
                param_grid=params, 
                scoring="accuracy", 
                cv=5, 
            )
            grid_adaboost.fit(X, y)
            
            st.success("Busca concluída!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Melhores Parâmetros", str(grid_adaboost.best_params_))
            with col2:
                st.metric("Melhor Acurácia", f"{grid_adaboost.best_score_:.3f}")
            
            # Mostrar resultados completos
            st.subheader("Resultados Completos da Busca em Grade")
            results_df = pd.DataFrame(grid_adaboost.cv_results_)
            st.dataframe(results_df[['param_n_estimators', 'param_learning_rate', 'mean_test_score', 'std_test_score']])

# Informações adicionais na sidebar
st.sidebar.markdown("---")
st.sidebar.header("Sobre o AdaBoost")
st.sidebar.info("""
AdaBoost (Adaptive Boosting) é um algoritmo de ensemble que combina 
múltiplos classificadores fracos para criar um classificador forte.

**Características:**
- Combina weak learners sequencialmente
- Ajusta pesos das instâncias
- Atribui pesos aos classificadores
""")