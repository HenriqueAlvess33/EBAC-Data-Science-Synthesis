import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Configuração da página
st.set_page_config(page_title="PCA - HAR Dataset", layout="wide")

# Título da aplicação
st.title("🔍 PCA - Análise de Componentes Principais")
st.markdown("""
Análise do dataset **Human Activity Recognition (HAR)** utilizando Árvores de Decisão e PCA.
""")

# Sidebar para configurações
st.sidebar.header("Configurações do Modelo")

# Upload dos arquivos ou usar dados de exemplo
st.sidebar.subheader("Carregamento de Dados")
use_sample_data = st.sidebar.checkbox("Usar dados de exemplo (simulados)", value=True)

if not use_sample_data:
    uploaded_files = {}
    file_types = {
        'features': 'features.txt',
        'labels': 'activity_labels.txt', 
        'subject_train': 'subject_train.txt',
        'X_train': 'X_train.txt',
        'y_train': 'y_train.txt',
        'subject_test': 'subject_test.txt',
        'X_test': 'X_test.txt',
        'y_test': 'y_test.txt'
    }
    
    for key, filename in file_types.items():
        uploaded_files[key] = st.sidebar.file_uploader(f"{filename}", type=['txt'])

# Parâmetros do modelo
st.sidebar.subheader("Parâmetros do Modelo")
ccp_alpha = st.sidebar.slider("ccp_alpha", min_value=0.001, max_value=0.01, value=0.001, step=0.001)
test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random State", min_value=0, max_value=10000, value=4500)

# Função para gerar dados simulados
def generate_sample_data():
    """Gera dados simulados para demonstração"""
    np.random.seed(4500)
    
    # Gerar features simuladas (561 features como no dataset original)
    n_samples_train = 7352
    n_samples_test = 2947
    n_features = 561
    
    X_train = np.random.randn(n_samples_train, n_features)
    X_test = np.random.randn(n_samples_test, n_features)
    
    # Gerar labels simulados (6 atividades)
    y_train = np.random.randint(1, 7, n_samples_train)
    y_test = np.random.randint(1, 7, n_samples_test)
    
    # Criar nomes de features simuladas
    features = [f'feature_{i:03d}' for i in range(n_features)]
    
    # Criar DataFrames
    X_train_df = pd.DataFrame(X_train, columns=features)
    X_test_df = pd.DataFrame(X_test, columns=features)
    y_train_df = pd.DataFrame(y_train, columns=['cod_label'])
    y_test_df = pd.DataFrame(y_test, columns=['cod_label'])
    
    return X_train_df, X_test_df, y_train_df, y_test_df, features

# Carregar dados
if use_sample_data:
    X_train, X_test, y_train, y_test, features = generate_sample_data()
    st.info("📊 Usando dados simulados para demonstração")
else:
    # Aqui você implementaria o carregamento dos arquivos reais
    st.warning("⚠️ Funcionalidade de upload de arquivos em desenvolvimento")
    X_train, X_test, y_train, y_test, features = generate_sample_data()

# Divisão dos dados
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train, y_train, test_size=test_size, random_state=random_state
)

# Layout em abas
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Árvore Completa", 
    "🔍 PCA - 1 Componente", 
    "📊 Múltiplos Componentes", 
    "📋 Conclusões"
])

with tab1:
    st.header("Árvore de Decisão com Todas as Variáveis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parâmetros")
        st.write(f"- **ccp_alpha**: {ccp_alpha}")
        st.write(f"- **Test Size**: {test_size}")
        st.write(f"- **Random State**: {random_state}")
        st.write(f"- **Número de Features**: {X_train.shape[1]}")
    
    with col2:
        st.subheader("Execução")
        if st.button("Executar Árvore Completa", key="tree_full"):
            with st.spinner("Treinando árvore de decisão..."):
                start_time = time.time()
                
                clf = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alpha)
                clf.fit(X_train_split, y_train_split.values.ravel())
                
                train_accuracy = accuracy_score(y_train_split, clf.predict(X_train_split))
                test_accuracy = accuracy_score(y_test_split, clf.predict(X_test_split))
                
                end_time = time.time()
                processing_time = end_time - start_time
            
            st.success("✅ Treinamento concluído!")
            st.metric("Acurácia - Treino", f"{train_accuracy:.3f}")
            st.metric("Acurácia - Teste", f"{test_accuracy:.3f}")
            st.metric("Tempo de Processamento", f"{processing_time:.2f} segundos")

with tab2:
    st.header("Árvore com PCA - 1 Componente")
    
    if st.button("Executar PCA com 1 Componente", key="pca_1"):
        with st.spinner("Aplicando PCA e treinando modelo..."):
            start_time = time.time()
            
            # Aplicar PCA
            prcomp = PCA(n_components=561).fit(X_train_split)
            X_train_pca = prcomp.transform(X_train_split)
            X_test_pca = prcomp.transform(X_test_split)
            
            # Usar apenas 1 componente
            n_components = 1
            pc_train = pd.DataFrame(X_train_pca[:, :n_components], columns=['cp1'])
            pc_test = pd.DataFrame(X_test_pca[:, :n_components], columns=['cp1'])
            
            # Treinar árvore
            clf = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alpha)
            clf.fit(pc_train, y_train_split.values.ravel())
            
            train_accuracy = accuracy_score(y_train_split, clf.predict(pc_train))
            test_accuracy = accuracy_score(y_test_split, clf.predict(pc_test))
            
            end_time = time.time()
            processing_time = end_time - start_time
        
        st.success("✅ PCA e treinamento concluídos!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Acurácia - Treino", f"{train_accuracy:.3f}")
            st.metric("Acurácia - Teste", f"{test_accuracy:.3f}")
            st.metric("Tempo de Processamento", f"{processing_time:.2f} segundos")
        
        with col2:
            st.subheader("Variância Explicada")
            explained_variance = prcomp.explained_variance_ratio_[0]
            st.metric("Variância explicada pelo CP1", f"{explained_variance:.3f}")
            
            # Gráfico do primeiro componente
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(pc_train['cp1'], [0] * len(pc_train), alpha=0.6)
            ax.set_xlabel('Primeiro Componente Principal (CP1)')
            ax.set_title('Distribuição do CP1')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

with tab3:
    st.header("Testando Múltiplos Componentes PCA")
    
    st.subheader("Configuração do Grid Search")
    n_components_list = st.multiselect(
        "Número de Componentes PCA para testar",
        options=[1, 2, 5, 10, 20, 50],
        default=[1, 2, 5, 10, 50]
    )
    
    cv_folds = st.slider("Número de folds na validação cruzada", min_value=5, max_value=15, value=10)
    
    if st.button("Executar Grid Search", key="grid_search"):
        if not n_components_list:
            st.error("Selecione pelo menos um número de componentes para testar.")
        else:
            with st.spinner("Executando Grid Search com validação cruzada..."):
                start_time = time.time()
                
                clf = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alpha)
                
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA()),
                    ('clf', clf)
                ])
                
                grid_params = {
                    'pca__n_components': n_components_list
                }
                
                grid = GridSearchCV(pipe, grid_params, cv=cv_folds, scoring='accuracy', verbose=0)
                grid.fit(X_train_split, y_train_split.values.ravel())
                
                y_pred = grid.predict(X_test_split)
                test_accuracy = accuracy_score(y_test_split, y_pred)
                train_accuracy = accuracy_score(y_train_split, grid.predict(X_train_split))
                
                end_time = time.time()
                processing_time = end_time - start_time
            
            st.success("✅ Grid Search concluído!")
            
            # Resultados
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Melhor Número de Componentes", grid.best_params_['pca__n_components'])
            
            with col2:
                st.metric("Acurácia - Teste", f"{test_accuracy:.3f}")
            
            with col3:
                st.metric("Tempo Total", f"{processing_time:.2f} segundos")
            
            # Tabela de resultados
            st.subheader("Resultados Detalhados")
            resultados = pd.DataFrame(grid.cv_results_)
            
            # Selecionar colunas relevantes
            cols_to_show = [
                'param_pca__n_components', 'mean_test_score', 'std_test_score', 
                'mean_fit_time'
            ]
            resultados_display = resultados[cols_to_show].copy()
            resultados_display.columns = [
                'N Componentes', 'Acurácia Média', 'Desvio Padrão', 'Tempo Médio (s)'
            ]
            resultados_display = resultados_display.round(4)
            
            st.dataframe(resultados_display, use_container_width=True)
            
            # Gráfico de comparação
            st.subheader("Comparação de Desempenho")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Gráfico de acurácia
            ax1.plot(
                resultados['param_pca__n_components'].astype(int),
                resultados['mean_test_score'],
                marker='o',
                linewidth=2,
                markersize=8
            )
            ax1.set_xlabel('Número de Componentes PCA')
            ax1.set_ylabel('Acurácia Média')
            ax1.set_title('Acurácia vs Número de Componentes')
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de tempo
            ax2.plot(
                resultados['param_pca__n_components'].astype(int),
                resultados['mean_fit_time'],
                marker='s',
                color='red',
                linewidth=2,
                markersize=8
            )
            ax2.set_xlabel('Número de Componentes PCA')
            ax2.set_ylabel('Tempo Médio de Treino (s)')
            ax2.set_title('Tempo vs Número de Componentes')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

with tab4:
    st.header("Conclusões")
    
    st.subheader("📊 Resumo dos Resultados")
    
    st.markdown("""
    ### O que aconteceu com a acurácia?
    - A acurácia demonstrou capacidade de se equiparar à árvore com todas as variáveis
    - Em muitos casos, é possível obter resultados próximos utilizando apenas uma fração das variáveis originais
    - Com poucos componentes principais (5-10), já é possível capturar a maior parte da informação relevante
    
    ### O que aconteceu com o tempo de processamento?
    - O PCA reduz significativamente o tempo de treinamento ao diminuir a dimensionalidade
    - Porém, o Grid Search com múltiplos componentes pode ser computacionalmente intensivo
    - O trade-off entre acurácia e tempo deve ser considerado conforme a aplicação
    
    ### Insights Importantes:
    - ✅ **Redução de dimensionalidade**: PCA permite trabalhar com menos variáveis mantendo boa performance
    - ✅ **Velocidade**: Menos componentes = treinamento mais rápido
    - ✅ **Generalização**: Menos chance de overfitting com menos variáveis
    - ⚠️ **Complexidade**: Grid Search adiciona overhead computacional
    """)
    
    st.info("""
    **Recomendação**: Para este dataset, testar com 5-10 componentes principais geralmente 
    oferece um bom equilíbrio entre performance e eficiência computacional.
    """)

# Rodapé
st.markdown("---")
st.markdown("**PCA - Human Activity Recognition Dataset** | Adaptado para Streamlit")