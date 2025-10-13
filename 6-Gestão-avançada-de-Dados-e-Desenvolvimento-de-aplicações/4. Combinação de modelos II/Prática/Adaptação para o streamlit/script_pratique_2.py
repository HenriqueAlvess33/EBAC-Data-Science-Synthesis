import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1, make_hastie_10_2, load_iris
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import tempfile
import os
import platform

# Configuração da página
st.set_page_config(page_title="GBM Analysis", layout="wide")

# Título principal
st.title("📊 Análise de Gradient Boosting Machines (GBM)")

# Função para detectar automaticamente o Tesseract
def find_tesseract():
    """Tenta encontrar o caminho do Tesseract automaticamente"""
    system = platform.system()
    
    if system == "Windows":
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
    elif system == "Linux":
        possible_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
        ]
    elif system == "Darwin":  # macOS
        possible_paths = [
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',
            '/usr/bin/tesseract',
        ]
    else:
        return None
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# Sidebar para navegação
st.sidebar.title("Navegação")
section = st.sidebar.radio("Selecione a seção:", [
    "OCR de PDF",
    "Diferenças AdaBoost vs GBM",
    "Exemplos GBM - Classificação e Regressão",
    "Hiperparâmetros do GBM",
    "GridSearch com Iris Dataset",
    "Stochastic GBM"
])

# Seção 1: OCR de PDF
if section == "OCR de PDF":
    st.header("🔍 OCR de PDF")
    
    # Configuração do Tesseract
    st.subheader("Configuração do Tesseract")
    
    # Tentar encontrar automaticamente
    auto_tesseract_path = find_tesseract()
    
    if auto_tesseract_path:
        st.success(f"Tesseract encontrado automaticamente: `{auto_tesseract_path}`")
        tesseract_path = auto_tesseract_path
    else:
        st.warning("⚠️ Tesseract não encontrado automaticamente.")
        
        # Instruções de instalação
        with st.expander("📋 Como instalar o Tesseract"):
            st.markdown("""
            ### Windows:
            1. Baixe o instalador do [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
            2. Instale em `C:\\Program Files\\Tesseract-OCR\\`
            3. Adicione ao PATH ou configure o caminho manualmente
            
            ### Linux (Ubuntu/Debian):
            ```bash
            sudo apt update
            sudo apt install tesseract-ocr
            sudo apt install tesseract-ocr-por  # Para português
            sudo apt install tesseract-ocr-eng  # Para inglês
            ```
            
            ### macOS:
            ```bash
            brew install tesseract
            ```
            """)
        
        tesseract_path = st.text_input(
            "Digite o caminho completo do executável do Tesseract:",
            placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
    
    if tesseract_path:
        try:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            # Testar se o Tesseract funciona
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                # Criar uma imagem de teste simples
                img = Image.new('RGB', (100, 50), color='white')
                img.save(tmp_file.name)
                
                # Testar OCR
                test_text = pytesseract.image_to_string(img)
                st.success("✅ Tesseract configurado com sucesso!")
                
        except Exception as e:
            st.error(f"❌ Erro ao configurar Tesseract: {e}")
            st.info("Verifique se o caminho está correto e o Tesseract está instalado.")
            tesseract_path = None
    
    # Upload do PDF (só mostra se Tesseract estiver configurado)
    if tesseract_path:
        uploaded_file = st.file_uploader("Faça upload de um PDF", type="pdf")
        
        if uploaded_file is not None:
            # Mostrar informações do arquivo
            file_details = {
                "Nome do arquivo": uploaded_file.name,
                "Tipo do arquivo": uploaded_file.type,
                "Tamanho do arquivo": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.write("**Informações do arquivo:**")
            st.json(file_details)
            
            # Configurações de OCR
            col1, col2 = st.columns(2)
            with col1:
                linguagem = st.selectbox(
                    "Idioma para OCR:",
                    ["eng", "por", "eng+por", "spa", "fra"],
                    index=0
                )
            with col2:
                dpi = st.slider("DPI para conversão:", min_value=150, max_value=400, value=200)
            
            if st.button("Executar OCR no PDF"):
                # Salvar o arquivo temporariamente
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name
                
                try:
                    # Converter PDF para imagens
                    with st.spinner('Convertendo PDF para imagens...'):
                        paginas = convert_from_path(pdf_path, dpi=dpi)
                    
                    st.success(f"✅ PDF convertido! {len(paginas)} páginas encontradas.")
                    
                    # Barra de progresso
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Executar OCR em cada página
                    todos_textos = []
                    
                    for i, imagem in enumerate(paginas):
                        status_text.text(f"Processando página {i+1} de {len(paginas)}...")
                        
                        # Realizar OCR
                        texto = pytesseract.image_to_string(imagem, lang=linguagem)
                        todos_textos.append({
                            "pagina": i + 1,
                            "texto": texto
                        })
                        
                        # Atualizar progresso
                        progress_bar.progress((i + 1) / len(paginas))
                    
                    status_text.text("✅ OCR concluído!")
                    
                    # Mostrar resultados
                    st.subheader("📄 Resultados do OCR")
                    
                    for resultado in todos_textos:
                        with st.expander(f"Página {resultado['pagina']} - {len(resultado['texto'])} caracteres"):
                            st.text_area(
                                f"Texto da página {resultado['pagina']}",
                                resultado['texto'],
                                height=200,
                                key=f"pagina_{resultado['pagina']}"
                            )
                    
                    # Opção para baixar todos os textos
                    texto_completo = "\n\n".join([f"--- Página {r['pagina']} ---\n{r['texto']}" for r in todos_textos])
                    
                    st.download_button(
                        label="📥 Baixar todos os textos",
                        data=texto_completo,
                        file_name=f"ocr_resultado_{uploaded_file.name.split('.')[0]}.txt",
                        mime="text/plain"
                    )
                    
                    # Limpar arquivo temporário
                    os.unlink(pdf_path)
                    
                except Exception as e:
                    st.error(f"❌ Erro ao processar PDF: {e}")
                    st.info("""
                    **Possíveis soluções:**
                    - Verifique se o PDF não está protegido/corrompido
                    - Tente reduzir o DPI
                    - Verifique as permissões do arquivo
                    """)
                    
                    # Limpar arquivo temporário em caso de erro
                    try:
                        os.unlink(pdf_path)
                    except:
                        pass
    else:
        st.info("👆 Configure o Tesseract acima para habilitar o OCR de PDFs.")

# Seção 2: Diferenças AdaBoost vs GBM
elif section == "Diferenças AdaBoost vs GBM":
    st.header("🤔 Diferenças entre AdaBoost e GBM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 AdaBoost")
        st.markdown("""
        - **Base**: Stumps (árvores de 1 nível)
        - **Pesos**: Modelos têm pesos diferentes
        - **Correção**: Foca em exemplos classificados erroneamente
        - **Amostragem**: Com reposição, baseada em pesos
        - **Função de perda**: Exponencial (classificação)
        """)
    
    with col2:
        st.subheader("🚀 Gradient Boosting")
        st.markdown("""
        - **Base**: Árvores com maior profundidade
        - **Pesos**: Mesmo peso para todos os modelos
        - **Correção**: Minimiza resíduos do modelo anterior
        - **Amostragem**: Pode usar subamostragem aleatória
        - **Função de perda**: Diversas opções disponíveis
        """)
    
    # Tabela comparativa
    st.subheader("📊 Comparação Detalhada")
    
    comparacao = pd.DataFrame({
        'Característica': [
            'Modelo Base',
            'Profundidade das Árvores',
            'Peso dos Modelos',
            'Estratégia de Correção',
            'Amostragem',
            'Funções de Perda'
        ],
        'AdaBoost': [
            'Stumps (árvores rasas)',
            'Muito rasa (1 nível)',
            'Pesos diferentes por modelo',
            'Foca em exemplos errados',
            'Com reposição, baseada em pesos',
            'Limitada (exponencial)'
        ],
        'GBM': [
            'Árvores de decisão',
            'Mais profundas',
            'Mesmo peso (× learning_rate)',
            'Minimiza resíduos',
            'Sem reposição, aleatória',
            'Diversas opções'
        ]
    })
    
    st.dataframe(comparacao, hide_index=True, use_container_width=True)

# Seção 3: Exemplos GBM
elif section == "Exemplos GBM - Classificação e Regressão":
    st.header("🧪 Exemplos Práticos do GBM")
    
    tab1, tab2 = st.tabs(["Classificação", "Regressão"])
    
    with tab1:
        st.subheader("Classificação Binária")
        st.markdown("Usando dataset `make_hastie_10_2` - problema de classificação binária sintético")
        
        if st.button("Executar Classificação", key="class_btn"):
            with st.spinner('Treinando modelo de classificação...'):
                X, y = make_hastie_10_2(random_state=0)
                X_train, X_test = X[:2000], X[2000:]
                y_train, y_test = y[:2000], y[2000:]
                
                clf = GradientBoostingClassifier(
                    n_estimators=100, 
                    learning_rate=1.0,
                    max_depth=1, 
                    random_state=0
                ).fit(X_train, y_train)
                
                accuracy = clf.score(X_test, y_test)
                
                # Resultados
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Acurácia do Modelo", f"{accuracy:.4f}")
                with col2:
                    st.metric("Número de Árvores", clf.n_estimators_)
                with col3:
                    st.metric("Learning Rate", "1.0")
                
                # Informações do dataset
                with st.expander("📋 Informações do Dataset"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Shape dos dados:**")
                        st.info(f"X: {X.shape}")
                        st.info(f"y: {y.shape}")
                        st.info(f"Classes únicas: {np.unique(y)}")
                    
                    with col2:
                        st.write("**Divisão treino/teste:**")
                        st.info(f"Treino: {X_train.shape[0]} amostras")
                        st.info(f"Teste: {X_test.shape[0]} amostras")
                        st.info(f"Features: {X.shape[1]}")
    
    with tab2:
        st.subheader("Regressão")
        st.markdown("Usando dataset `make_friedman1` - problema de regressão sintético")
        
        if st.button("Executar Regressão", key="reg_btn"):
            with st.spinner('Treinando modelo de regressão...'):
                X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
                X_train, X_test = X[:200], X[200:]
                y_train, y_test = y[:200], y[200:]
                
                est = GradientBoostingRegressor(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=1, 
                    random_state=0,
                    loss='squared_error'
                ).fit(X_train, y_train)
                
                mse = mean_squared_error(y_test, est.predict(X_test))
                rmse = np.sqrt(mse)
                
                # Resultados
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col3:
                    st.metric("R² Score", f"{est.score(X_test, y_test):.4f}")
                
                # Visualização das previsões
                with st.expander("🔍 Detalhes das Previsões"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Primeiras 10 previsões:**")
                        preview_df = pd.DataFrame({
                            'Real': y_test[:10],
                            'Previsto': est.predict(X_test)[:10],
                            'Erro': np.abs(y_test[:10] - est.predict(X_test)[:10])
                        })
                        st.dataframe(preview_df.style.format("{:.4f}"))
                    
                    with col2:
                        st.write("**Estatísticas do modelo:**")
                        st.info(f"Número de árvores: {est.n_estimators_}")
                        st.info(f"Features: {X.shape[1]}")
                        st.info(f"Loss function: {est.loss}")

# Seção 4: Hiperparâmetros
elif section == "Hiperparâmetros do GBM":
    st.header("⚙️ Hiperparâmetros Importantes do GBM")
    
    # Explicação interativa
    parametro = st.selectbox(
        "Selecione um hiperparâmetro para detalhes:",
        [
            "n_estimators",
            "learning_rate", 
            "max_depth",
            "subsample",
            "max_features",
            "min_samples_split",
            "min_samples_leaf"
        ]
    )
    
    explicacoes = {
        "n_estimators": {
            "descricao": "Número de árvores no ensemble",
            "impacto": "Mais árvores geralmente melhoram performance mas aumentam tempo de treino",
            "valores_tipicos": "50-500",
            "dica": "Use early stopping para encontrar o número ideal"
        },
        "learning_rate": {
            "descricao": "Taxa de aprendizado - controla contribuição de cada árvore",
            "impacto": "Valores menores exigem mais árvores mas melhoram generalização",
            "valores_tipicos": "0.01-0.3",
            "dica": "Reduza learning_rate e aumente n_estimators para melhor performance"
        },
        "max_depth": {
            "descricao": "Profundidade máxima de cada árvore",
            "impacto": "Árvores mais profundas capturam padrões complexos mas podem overfittar",
            "valores_tipicos": "3-10",
            "dica": "Comece com 3-6 para problemas simples"
        },
        "subsample": {
            "descricao": "Fraçao de amostras usadas para treinar cada árvore",
            "impacto": "Valores < 1.0 introduzem randomização que reduz overfitting",
            "valores_tipicos": "0.7-1.0", 
            "dica": "Use 0.8 para começar - cria Stochastic GBM"
        },
        "max_features": {
            "descricao": "Número máximo de features consideradas por split",
            "impacto": "Reduz correlação entre árvores, melhora generalização",
            "valores_tipicos": "sqrt(n_features) ou log2(n_features)",
            "dica": "Use 'sqrt' para classificação, 'log2' para regressão"
        },
        "min_samples_split": {
            "descricao": "Número mínimo de amostras para dividir um nó",
            "impacto": "Controla a complexidade das árvores",
            "valores_tipicos": "2-20",
            "dica": "Valores maiores previnem overfitting"
        },
        "min_samples_leaf": {
            "descricao": "Número mínimo de amostras em uma folha",
            "impacto": "Folhas muito pequenas podem capturar ruído",
            "valores_tipicos": "1-10", 
            "dica": "Use valores maiores para dados com ruído"
        }
    }
    
    if parametro in explicacoes:
        info = explicacoes[parametro]
        st.subheader(f"`{parametro}`")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Descrição:**")
            st.info(info["descricao"])
            
            st.write("**Impacto:**")
            st.info(info["impacto"])
        
        with col2:
            st.write("**Valores típicos:**")
            st.info(info["valores_tipicos"])
            
            st.write("**Dica prática:**")
            st.info(info["dica"])
    
    # Guia rápido
    st.subheader("🎯 Guia Rápido de Configuração")
    
    st.markdown("""
    **Para começar rapidamente:**
    ```python
    # Classificação
    GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    
    # Regressão  
    GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1, 
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    ```
    """)

# Seção 5: GridSearch
elif section == "GridSearch com Iris Dataset":
    st.header("🔎 Otimização com GridSearchCV")
    
    st.subheader("Dataset Iris")
    X, y = load_iris(return_X_y=True)
    df = load_iris()
    
    # Mostrar informações do dataset
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Primeiras 5 linhas:**")
        preview_df = pd.DataFrame(X[:5], columns=df.feature_names)
        preview_df['target'] = y[:5]
        preview_df['target_name'] = [df.target_names[t] for t in y[:5]]
        st.dataframe(preview_df)
    
    with col2:
        st.write("**Informações do dataset:**")
        st.info(f"Shape: {X.shape}")
        st.info(f"Número de classes: {len(np.unique(y))}")
        st.info(f"Nomes das classes: {list(df.target_names)}")
        
        # Estatísticas básicas
        st.write("**Estatísticas:**")
        stats_df = pd.DataFrame(X, columns=df.feature_names).describe()
        st.dataframe(stats_df)
    
    # Configuração do GridSearch
    st.subheader("Configuração do GridSearch")
    
    with st.form("gridsearch_config"):
        st.write("Selecione os valores para testar:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_estimators = st.multiselect(
                "n_estimators",
                [50, 100, 150, 200],
                default=[100, 150]
            )
        
        with col2:
            learning_rate = st.multiselect(
                "learning_rate",
                [0.01, 0.05, 0.1, 0.2],
                default=[0.1, 0.2]
            )
        
        with col3:
            subsample = st.multiselect(
                "subsample",
                [0.6, 0.7, 0.8, 0.9, 1.0],
                default=[0.8, 1.0]
            )
        
        # Opções avançadas
        with st.expander("Opções Avançadas"):
            cv_folds = st.slider("Número de folds para validação cruzada:", 3, 10, 5)
            scoring = st.selectbox(
                "Métrica de avaliação:",
                ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
            )
        
        executar_gridsearch = st.form_submit_button("🚀 Executar GridSearch")
    
    if executar_gridsearch and n_estimators and learning_rate and subsample:
        with st.spinner('Executando GridSearch... Isso pode levar alguns minutos'):
            params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'subsample': subsample,
            }
            
            grid_clf = GridSearchCV(
                estimator=GradientBoostingClassifier(random_state=42),
                param_grid=params,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=True
            )
            grid_clf.fit(X, y)
            
            # Resultados
            st.subheader("🎯 Resultados do GridSearch")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Melhor Score", f"{grid_clf.best_score_:.4f}")
            with col2:
                st.metric("Número de Combinações", len(grid_clf.cv_results_['params']))
            with col3:
                st.metric("Melhor Estimador", f"GBM ({grid_clf.best_params_['n_estimators']} trees)")
            
            st.write("**Melhores Hiperparâmetros:**")
            for param, value in grid_clf.best_params_.items():
                st.success(f"`{param}: {value}`")
            
            # Mostrar todos os resultados em uma tabela
            st.subheader("📊 Todos os Resultados")
            results_df = pd.DataFrame(grid_clf.cv_results_)
            cols_to_show = ['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']
            results_display = results_df[cols_to_show].sort_values('mean_test_score', ascending=False)
            results_display['rank'] = range(1, len(results_display) + 1)
            
            st.dataframe(results_display.head(10), use_container_width=True)
            
            # Gráfico de comparação
            st.subheader("📈 Comparação dos Hiperparâmetros")
            
            # Criar visualização simples
            comparison_data = []
            for i, row in results_display.head(8).iterrows():
                comparison_data.append({
                    'Params': str(row['params'])[:50] + "...",
                    'Test Score': row['mean_test_score'],
                    'Train Score': row['mean_train_score']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.bar_chart(comparison_df.set_index('Params'))
    
    elif executar_gridsearch:
        st.warning("⚠️ Selecione pelo menos um valor para cada hiperparâmetro!")

# Seção 6: Stochastic GBM
elif section == "Stochastic GBM":
    st.header("🎲 Stochastic Gradient Boosting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 O que é Stochastic GBM?")
        st.markdown("""
        O **Stochastic GBM** introduz aleatoriedade no processo de boosting através de:
        
        - **Subamostragem dos dados** (`subsample < 1.0`)
        - **Subamostragem das features** (`max_features < 1.0`)
        - **Ambas as técnicas combinadas**
        
        Esta abordagem cria um ensemble mais diversificado e robusto.
        """)
        
        st.subheader("📊 Benefícios Comprovados")
        st.markdown("""
        - ✅ **Melhor generalização**
        - ✅ **Redução de overfitting**  
        - ✅ **Treinamento mais rápido**
        - ✅ **Maior diversidade entre árvores**
        - ✅ **Performance superior em dados ruidosos**
        """)
    
    with col2:
        st.subheader("🎛️ Como Configurar")
        
        subsample_val = st.slider("Subsample Rate", 0.1, 1.0, 0.8, 0.1)
        max_features_val = st.selectbox("Max Features", ["sqrt", "log2", "auto", 0.8, 0.9])
        
        st.code(f"""
        # Stochastic GBM Configuration
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            subsample={subsample_val},
            max_features='{max_features_val}',
            random_state=42
        )
        """)
        
        st.info(f"🔬 Cada árvore usará {subsample_val*100}% dos dados e features selecionadas")
    
    # Descobertas do artigo
    st.subheader("📚 Descobertas do Artigo de Friedman")
    
    st.markdown("""
    ### Principais Conclusões:
    
    - **Conjuntos grandes de dados**: Melhor performance com subamostras de **40-60%**
    - **Conjuntos pequenos**: Subamostras de **20-40%** são mais eficientes  
    - **Árvores profundas**: Beneficiam-se mais da randomização
    - **Dados ruidosos**: Subamostragem ajuda significativamente
    
    ### Recomendações Práticas:
    
    1. **Comece com `subsample=0.8`** para a maioria dos casos
    2. **Use `max_features='sqrt'`** para classificação
    3. **Combine com learning_rate baixo** (0.05-0.2)
    4. **Aumente n_estimators** para compensar a subamostragem
    """)
    
    # Comparação visual
    st.subheader("📈 Impacto da Subamostragem")
    
    data_comparison = pd.DataFrame({
        'Tamanho do Dataset': ["Pequeno (< 1K)", "Médio (1K-10K)", "Grande (> 10K)"],
        'Subsample Recomendado': ["20-40%", "40-60%", "50-80%"],
        'Benefício Esperado': ["Alto", "Muito Alto", "Moderado-Alto"]
    })
    
    st.dataframe(data_comparison, hide_index=True, use_container_width=True)

# Rodapé
st.sidebar.markdown("---")
st.sidebar.info("""
**Desenvolvido com Streamlit**  
📚 Análise de Gradient Boosting Machines  
⚡ Machine Learning
""")

# Adicionar informações de debug no sidebar
if st.sidebar.checkbox("Mostrar informações do sistema"):
    st.sidebar.write("**Sistema:**", platform.system())
    st.sidebar.write("**Tesseract encontrado:**", "✅ Sim" if find_tesseract() else "❌ Não")
    
    try:
        import sklearn
        st.sidebar.write("**Scikit-learn:**", sklearn.__version__)
    except:
        st.sidebar.write("**Scikit-learn:** ❌ Erro")