import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import io

# Configuração da página
st.set_page_config(page_title="Análise de Atividades Humanas", layout="wide")

st.title("Análise de Atividades Humanas - Classificação com Árvore de Decisão")

# Sidebar para upload de arquivos
st.sidebar.header("Upload dos Arquivos")

# Função para ler arquivos
def read_uploaded_file(file, delimiter=None, delim_whitespace=False, header=None, names=None):
    if delimiter:
        return pd.read_csv(file, delimiter=delimiter, header=header, names=names)
    elif delim_whitespace:
        return pd.read_csv(file, delim_whitespace=True, header=header, names=names)
    else:
        return pd.read_csv(file, header=header, names=names)

# Área de upload para cada arquivo necessário
features_file = st.sidebar.file_uploader("Features (features.txt)", type=['txt'])
subject_train_file = st.sidebar.file_uploader("Subject Train (subject_train.txt)", type=['txt'])
x_train_file = st.sidebar.file_uploader("X Train (X_train.txt)", type=['txt'])
y_train_file = st.sidebar.file_uploader("Y Train (y_train.txt)", type=['txt'])
subject_test_file = st.sidebar.file_uploader("Subject Test (subject_test.txt)", type=['txt'])
x_test_file = st.sidebar.file_uploader("X Test (X_test.txt)", type=['txt'])
y_test_file = st.sidebar.file_uploader("Y Test (y_test.txt)", type=['txt'])

# Verifica se todos os arquivos foram carregados
required_files = [features_file, subject_train_file, x_train_file, y_train_file, 
                 subject_test_file, x_test_file, y_test_file]

if all(required_files):
    
    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Carregando arquivos...")
        
        # Carrega os arquivos
        features = read_uploaded_file(features_file, delimiter="\t", header=None)[0]
        progress_bar.progress(15)
        
        subject_train = read_uploaded_file(subject_train_file, delimiter="\t", header=None)[0]
        progress_bar.progress(25)
        
        x_train = read_uploaded_file(x_train_file, delim_whitespace=True, header=None, names=features)
        progress_bar.progress(35)
        
        y_train = read_uploaded_file(y_train_file, delim_whitespace=True, header=None)
        progress_bar.progress(45)
        
        subject_test = read_uploaded_file(subject_test_file, delimiter="\t", header=None)[0]
        progress_bar.progress(55)
        
        x_test = read_uploaded_file(x_test_file, delim_whitespace=True, header=None, names=features)
        progress_bar.progress(65)
        
        y_test = read_uploaded_file(y_test_file, delim_whitespace=True, header=None)
        progress_bar.progress(75)
        
        status_text.text("Processando dados...")
        
        # Cria os DataFrames
        dataframe_treino = (
            pd.DataFrame(x_train)
            .assign(Indivíduos=subject_train)
            .assign(Atividades=y_train)
            .set_index("Indivíduos", append=True)
        )
        
        dataframe_teste = (
            pd.DataFrame(x_test)
            .assign(Indivíduos=subject_test)
            .assign(Atividades=y_test)
            .set_index("Indivíduos", append=True)
        )
        
        progress_bar.progress(85)
        
        # Concatena os DataFrames
        df1 = pd.concat([dataframe_treino, dataframe_teste], axis=0, ignore_index=True)
        
        # Preparação dos dados
        X = df1.drop(columns=["Atividades"])
        y = df1["Atividades"]
        
        # Divisão dos dados
        X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.2, random_state=2360873)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_, y_, test_size=0.25, random_state=2360873
        )
        
        progress_bar.progress(95)
        
        # Treinamento do modelo inicial
        clf0 = DecisionTreeClassifier(max_depth=4, random_state=2510895)
        clf0.fit(X_train, y_train)
        
        # Análise de importância das variáveis
        colunas = X.columns
        importancia = clf0.feature_importances_
        df_top3_var = pd.DataFrame({"Colunas": colunas, "variaveis": importancia}).sort_values(
            by="variaveis", ascending=False
        )
        
        progress_bar.progress(100)
        status_text.text("Processamento concluído!")
        
        # Exibe resultados
        st.header("Resultados da Análise")
        
        # Mostra as variáveis mais importantes
        st.subheader("Top 10 Variáveis Mais Importantes")
        st.dataframe(df_top3_var.head(10))
        
        # Seleciona as 3 melhores variáveis
        filtro_de_colunas = df_top3_var.head(3)['Colunas'].tolist()
        st.write(f"Variáveis selecionadas para o modelo final: {filtro_de_colunas}")
        
        # Reorganiza os dados com as melhores variáveis
        X = df1[filtro_de_colunas]
        
        X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.2, random_state=2360873)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_, y_, test_size=0.25, random_state=2360873
        )
        
        # Otimização do parâmetro alpha
        st.subheader("Otimização do Parâmetro Alpha")
        
        caminho = DecisionTreeClassifier(random_state=2360873).cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = caminho.ccp_alphas, caminho.impurities
        
        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=2360873, ccp_alpha=ccp_alpha).fit(X_train, y_train)
            clfs.append(clf)
            
        train_scores = [clf.score(X_train, y_train) for clf in clfs]
        valid_scores = [clf.score(X_valid, y_valid) for clf in clfs]
        
        # Gráfico de Acurácia x Alpha
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Acurácia")
        ax.set_title("Acurácia x Alpha do conjunto de dados de treino e validação")
        ax.plot(ccp_alphas, train_scores, marker='o', label="Treino", drawstyle="steps-post")
        ax.plot(ccp_alphas, valid_scores, marker='o', label="Validação", drawstyle="steps-post")
        ax.legend()
        st.pyplot(fig)
        
        # Encontra a melhor árvore
        ind_melhor_arvore = len(valid_scores) - valid_scores[::-1].index(max(valid_scores)) - 1
        melhor_arvore = clfs[ind_melhor_arvore]
        
        st.metric(
            label="Acurácia da Melhor Árvore (Validação)",
            value=f"{valid_scores[ind_melhor_arvore]*100:.2f}%"
        )
        
        # Avaliação no conjunto de teste
        acuracia_teste = melhor_arvore.score(X_test, y_test)
        
        st.metric(
            label="Acurácia Final (Teste)",
            value=f"{acuracia_teste*100:.2f}%"
        )
        
        # Matriz de Confusão
        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_test, melhor_arvore.predict(X_test))
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax_cm)
        st.pyplot(fig_cm)
        
        # Informações adicionais
        st.subheader("Informações do Modelo")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Parâmetros da Melhor Árvore:**")
            st.write(f"- Profundidade máxima: {melhor_arvore.get_depth()}")
            st.write(f"- Número de folhas: {melhor_arvore.get_n_leaves()}")
            st.write(f"- Alpha utilizado: {ccp_alphas[ind_melhor_arvore]:.6f}")
        
        with col2:
            st.write("**Estatísticas dos Dados:**")
            st.write(f"- Total de amostras: {len(df1)}")
            st.write(f"- Amostras de treino: {len(X_train)}")
            st.write(f"- Amostras de validação: {len(X_valid)}")
            st.write(f"- Amostras de teste: {len(X_test)}")
        
    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {str(e)}")
        st.info("Verifique se os arquivos estão no formato correto.")

else:
    st.info("""
    ## Instruções de Uso:
    
    1. **Faça o upload de todos os arquivos necessários** na barra lateral
    2. **Arquivos necessários:**
       - features.txt
       - subject_train.txt
       - X_train.txt
       - y_train.txt
       - subject_test.txt
       - X_test.txt
       - y_test.txt
    
    3. **Formato esperado:** Arquivos de texto com dados de sensores de atividades humanas
    
    O sistema irá automaticamente:
    - Processar os dados
    - Identificar as variáveis mais importantes
    - Treinar um modelo de árvore de decisão
    - Otimizar os parâmetros
    - Mostrar os resultados e métricas
    """)

# Rodapé
st.sidebar.markdown("---")
st.sidebar.info(
    "Sistema de análise de atividades humanas usando Machine Learning. "
    "Carregue todos os arquivos necessários para iniciar a análise."
)