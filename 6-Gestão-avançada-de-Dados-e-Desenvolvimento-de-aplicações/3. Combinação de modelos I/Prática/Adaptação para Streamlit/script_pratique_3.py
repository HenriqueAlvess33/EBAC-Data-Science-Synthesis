import streamlit as st

st.set_page_config(page_title="Hyperparâmetros do Random Forest", layout="wide")

st.title("🌳 Hyperparâmetros do Random Forest")
st.markdown("---")

# Introdução
st.header("📋 Introdução")
st.write("""
O Random Forest (Floresta Aleatória) é um algoritmo de ensemble que combina múltiplas árvores de decisão.
Cada árvore é treinada em subconjuntos diferentes dos dados, criando um modelo robusto e menos propenso a overfitting.
""")

# Seção de Hyperparâmetros
st.header("⚙️ Principais Hyperparâmetros")

# Criando abas para organizar os parâmetros
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Parâmetros Básicos", 
    "🌳 Controle de Árvores", 
    "🔧 Configurações Avançadas", 
    "⚡ Performance"
])

with tab1:
    st.subheader("Parâmetros Fundamentais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### n_estimators")
        st.write("**Número de árvores na floresta**")
        st.write("""
        - **Valores típicos**: 50-500
        - **Impacto**: Mais árvores geralmente melhoram a performance, mas aumentam o tempo de treinamento
        - **Padrão**: 100
        """)
        
        st.markdown("#### criterion")
        st.write("**Função para medir a qualidade das divisões**")
        
        criterion_options = ["gini", "entropy", "log_loss"]
        selected_criterion = st.selectbox("Escolha o criterion:", criterion_options)
        
        if selected_criterion == "gini":
            st.info("**Gini**: Impureza simples - Rápido e eficiente")
        elif selected_criterion == "entropy":
            st.info("**Entropy**: Baseado em informação - Mais preciso")
        else:
            st.info("**Log Loss**: Otimiza probabilidades preditas - Mais lento mas mais preciso")
    
    with col2:
        st.markdown("#### max_features")
        st.write("**Número de features consideradas em cada divisão**")
        
        max_feat_options = ["sqrt", "log2", "None", "Customizado"]
        selected_max_feat = st.selectbox("Estratégia max_features:", max_feat_options)
        
        if selected_max_feat == "sqrt":
            st.write("`max_features = sqrt(n_features)`")
        elif selected_max_feat == "log2":
            st.write("`max_features = log2(n_features)`")
        elif selected_max_feat == "None":
            st.write("`max_features = n_features` (todas as features)")
        else:
            custom_value = st.slider("Número de features:", 1, 50, 10)
            st.write(f"`max_features = {custom_value}`")

with tab2:
    st.subheader("Controle do Crescimento das Árvores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### max_depth")
        st.write("**Profundidade máxima das árvores**")
        depth_option = st.radio("max_depth:", ["None (ilimitado)", "Customizado"])
        if depth_option == "Customizado":
            custom_depth = st.slider("Profundidade máxima:", 1, 50, 10)
            st.write(f"`max_depth = {custom_depth}`")
        else:
            st.write("`max_depth = None`")
        
        st.markdown("#### min_samples_split")
        st.write("**Mínimo de amostras para dividir um nó**")
        min_split = st.slider("min_samples_split:", 2, 100, 2)
        st.write(f"`min_samples_split = {min_split}`")
    
    with col2:
        st.markdown("#### min_samples_leaf")
        st.write("**Mínimo de amostras por folha**")
        min_leaf = st.slider("min_samples_leaf:", 1, 50, 1)
        st.write(f"`min_samples_leaf = {min_leaf}`")
        
        st.markdown("#### max_leaf_nodes")
        st.write("**Número máximo de folhas**")
        leaf_nodes_option = st.radio("max_leaf_nodes:", ["None (ilimitado)", "Customizado"])
        if leaf_nodes_option == "Customizado":
            custom_nodes = st.slider("Número máximo de folhas:", 10, 1000, 100)
            st.write(f"`max_leaf_nodes = {custom_nodes}`")

with tab3:
    st.subheader("Configurações Avançadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### bootstrap")
        bootstrap = st.toggle("bootstrap", value=True)
        if bootstrap:
            st.write("✅ Amostragem com reposição ativada")
            st.write("Cada árvore treina em subconjunto diferente")
        else:
            st.write("❌ Todas as árvores usam dataset completo")
        
        st.markdown("#### oob_score")
        oob_score = st.toggle("oob_score", value=False)
        if oob_score and bootstrap:
            st.success("✅ Score out-of-bag será calculado")
        elif oob_score and not bootstrap:
            st.error("❌ oob_score requer bootstrap=True")
        
        st.markdown("#### random_state")
        random_state = st.number_input("random_state:", min_value=0, value=42)
        st.write(f"`random_state = {int(random_state)}`")
    
    with col2:
        st.markdown("#### class_weight")
        weight_options = ["None", "balanced", "balanced_subsample", "Customizado"]
        selected_weight = st.selectbox("Estratégia de pesos:", weight_options)
        
        if selected_weight == "balanced":
            st.write("Pesos calculados automaticamente baseado na frequência inversa")
        elif selected_weight == "balanced_subsample":
            st.write("Pesos calculados para cada subconjunto bootstrap")
        
        st.markdown("#### ccp_alpha")
        ccp_alpha = st.slider("ccp_alpha (poda por complexidade):", 0.0, 1.0, 0.0, 0.01)
        st.write(f"`ccp_alpha = {ccp_alpha}`")

with tab4:
    st.subheader("Configurações de Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### n_jobs")
        st.write("**Número de núcleos de CPU para paralelização**")
        
        jobs_options = {
            "1 núcleo": 1,
            "Todos os núcleos (-1)": -1,
            "2 núcleos": 2,
            "4 núcleos": 4,
            "8 núcleos": 8
        }
        
        selected_jobs = st.selectbox("n_jobs:", list(jobs_options.keys()))
        st.write(f"`n_jobs = {jobs_options[selected_jobs]}`")
    
    with col2:
        st.markdown("#### warm_start")
        warm_start = st.toggle("warm_start", value=False)
        if warm_start:
            st.write("✅ Permite adicionar mais árvores sem retreinar do zero")
        else:
            st.write("❌ Novo treinamento a cada chamada do fit()")
        
        st.markdown("#### verbose")
        verbose_level = st.slider("Nível de verbosidade:", 0, 3, 0)
        st.write(f"`verbose = {verbose_level}`")

# Resumo Interativo
st.markdown("---")
st.header("🎯 Resumo da Configuração")

if st.button("Gerar Código Python"):
    code = f"""
from sklearn.ensemble import RandomForestClassifier

# Configuração do Random Forest com seus parâmetros
model = RandomForestClassifier(
    n_estimators=100,
    criterion='{selected_criterion}',
    max_depth={'None' if depth_option.startswith('None') else custom_depth},
    min_samples_split={min_split},
    min_samples_leaf={min_leaf},
    max_features={'sqrt' if selected_max_feat == 'sqrt' else 
                 'log2' if selected_max_feat == 'log2' else 
                 None if selected_max_feat == 'None' else custom_value},
    bootstrap={bootstrap},
    oob_score={oob_score},
    n_jobs={jobs_options[selected_jobs]},
    random_state={int(random_state)},
    verbose={verbose_level},
    warm_start={warm_start},
    ccp_alpha={ccp_alpha}
)
"""
    st.code(code, language='python')

# Dicas e Melhores Práticas
st.markdown("---")
st.header("💡 Dicas e Melhores Práticas")

with st.expander("🔍 Como escolher os melhores parâmetros"):
    st.write("""
    - **Comece com valores padrão** e ajuste gradualmente
    - Use **GridSearchCV** ou **RandomizedSearchCV** para otimização
    - **n_estimators**: Aumente até a performance estabilizar
    - **max_depth**: Controle para evitar overfitting
    - **max_features**: Valores menores aumentam a diversidade das árvores
    """)

with st.expander("📊 Comparação dos Critérios"):
    st.table({
        "Criterion": ["gini", "entropy", "log_loss"],
        "Velocidade": ["Rápido", "Médio", "Mais lento"],
        "Precisão": ["Regular", "Boa", "Muito boa"],
        "Probabilidades": ["Regular", "Levemente melhor", "Muito melhor"]
    })

st.markdown("---")
st.info("💡 **Lembre-se**: A melhor combinação de parâmetros depende dos seus dados específicos. Experimente diferentes configurações!")