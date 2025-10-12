import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import os
import io

st.set_page_config(page_title="Análise de Árvore de Decisão", layout="wide")

st.title("📊 Análise de Classificação com Árvore de Decisão")
st.markdown("---")

# Área de upload de arquivos
st.header("📁 Upload dos Arquivos")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Arquivos de Treino")
    features_file = st.file_uploader("Features (features.txt)", type=['txt'], key='features')
    subject_train_file = st.file_uploader("Subject Train (subject_train.txt)", type=['txt'], key='subject_train')
    x_train_file = st.file_uploader("X Train (X_train.txt)", type=['txt'], key='x_train')
    y_train_file = st.file_uploader("Y Train (y_train.txt)", type=['txt'], key='y_train')

with col2:
    st.subheader("Arquivos de Teste")
    subject_test_file = st.file_uploader("Subject Test (subject_test.txt)", type=['txt'], key='subject_test')
    x_test_file = st.file_uploader("X Test (X_test.txt)", type=['txt'], key='x_test')
    y_test_file = st.file_uploader("Y Test (y_test.txt)", type=['txt'], key='y_test')

# Verifica se todos os arquivos foram carregados
arquivos_necessarios = [
    features_file, subject_train_file, x_train_file, y_train_file,
    subject_test_file, x_test_file, y_test_file
]

if all(arquivos_necessarios):
    try:
        # Carregando os dados
        st.header("📈 Processamento dos Dados")
        
        with st.spinner("Carregando dados..."):
            # Lendo os arquivos
            features = pd.read_csv(features_file, delimiter='\t', header=None)[0]
            subject_train = pd.read_csv(subject_train_file, delimiter='\t', header=None)[0]
            x_train = pd.read_csv(x_train_file, delim_whitespace=True, header=None, names=features)
            y_train = pd.read_csv(y_train_file, delim_whitespace=True, header=None)

            subject_test = pd.read_csv(subject_test_file, delimiter='\t', header=None)[0]
            x_test = pd.read_csv(x_test_file, delim_whitespace=True, header=None, names=features)
            y_test = pd.read_csv(y_test_file, delim_whitespace=True, header=None)

        # Criando dataframes
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

        # Aplicando filtro de colunas
        filtro_de_colunas = ['1 tBodyAcc-mean()-X', '2 tBodyAcc-mean()-Y', '3 tBodyAcc-mean()-Z', 'Atividades']
        
        # Verificando se as colunas existem nos dataframes
        colunas_existentes_treino = [col for col in filtro_de_colunas if col in dataframe_treino.columns]
        colunas_existentes_teste = [col for col in filtro_de_colunas if col in dataframe_teste.columns]
        
        dataframe_treino = dataframe_treino[colunas_existentes_treino]
        dataframe_teste = dataframe_teste[colunas_existentes_teste]

        # Mostrando preview dos dados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dados de Treino")
            st.dataframe(dataframe_treino.head(), use_container_width=True)
            st.write(f"Shape: {dataframe_treino.shape}")
            
        with col2:
            st.subheader("Dados de Teste")
            st.dataframe(dataframe_teste.head(), use_container_width=True)
            st.write(f"Shape: {dataframe_teste.shape}")

        # Preparando dados para o modelo
        x_train_model = dataframe_treino.drop(columns=['Atividades'])
        y_train_model = dataframe_treino['Atividades']
        x_test_model = dataframe_teste.drop(columns=['Atividades'])
        y_test_model = dataframe_teste['Atividades']

        st.header("🎯 Treinamento do Modelo")
        
        # Parâmetros do modelo
        min_samples_leaf = st.slider("min_samples_leaf", min_value=1, max_value=50, value=20)
        
        if st.button("Treinar Modelo"):
            with st.spinner("Treinando modelo..."):
                # Modelo inicial
                clf = DecisionTreeClassifier(random_state=100, min_samples_leaf=min_samples_leaf)
                clf.fit(x_train_model, y_train_model)

                # Path para pruning
                path = clf.cost_complexity_pruning_path(x_train_model, y_train_model)
                ccp_alphas, impurities = path.ccp_alphas, path.impurities

                # Treinando múltiplos modelos com diferentes alphas
                clfs = []
                for i in range(0, len(ccp_alphas), 5):
                    ccp_alpha = ccp_alphas[i]
                    clf = DecisionTreeClassifier(random_state=100, ccp_alpha=ccp_alpha, min_samples_leaf=min_samples_leaf)
                    clf.fit(x_train_model, y_train_model)
                    clfs.append(clf)
                
                ccp_alphas_reduzidos = ccp_alphas[::5]

                # Métricas MSE
                st.subheader("📉 MSE x Alpha")
                train_scores_mse = [mean_squared_error(y_train_model, clf.predict(x_train_model)) for clf in clfs]
                test_scores_mse = [mean_squared_error(y_test_model, clf.predict(x_test_model)) for clf in clfs]

                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.set_xlabel("Alpha")
                ax1.set_ylabel("MSE")
                ax1.set_title("MSE x Alpha do conjunto de dados de treino e teste")
                ax1.plot(ccp_alphas_reduzidos, train_scores_mse, marker="o", label="treino", drawstyle="steps-post")
                ax1.plot(ccp_alphas_reduzidos, test_scores_mse, marker="o", label="teste", drawstyle="steps-post")
                ax1.legend()
                st.pyplot(fig1)

                # Encontrando melhor árvore baseada no MSE
                ind_melhor_arvore_mse = len(test_scores_mse) - test_scores_mse[::-1].index(max(test_scores_mse)) - 1
                melhor_arvore_mse = clfs[ind_melhor_arvore_mse]

                # Métricas Acurácia
                st.subheader("📈 Acurácia x Alpha")
                train_scores_acc = [accuracy_score(y_train_model, clf.predict(x_train_model)) for clf in clfs]
                test_scores_acc = [accuracy_score(y_test_model, clf.predict(x_test_model)) for clf in clfs]

                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.set_xlabel("Alpha")
                ax2.set_ylabel("Acurácia")
                ax2.set_title("Acurácia x Alpha do conjunto de dados de treino e teste")
                ax2.plot(ccp_alphas_reduzidos, train_scores_acc, marker="o", label="treino", drawstyle="steps-post")
                ax2.plot(ccp_alphas_reduzidos, test_scores_acc, marker="o", label="teste", drawstyle="steps-post")
                ax2.legend()
                st.pyplot(fig2)

                # Encontrando melhor árvore baseada na acurácia
                ind_melhor_arvore_acc = len(test_scores_acc) - test_scores_acc[::-1].index(max(test_scores_acc)) - 1
                melhor_arvore_acc = clfs[ind_melhor_arvore_acc]

                # Resultados
                st.header("📊 Resultados Finais")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Melhor MSE (Teste)", f"{test_scores_mse[ind_melhor_arvore_mse]:.4f}")
                    st.metric("Alpha correspondente", f"{ccp_alphas_reduzidos[ind_melhor_arvore_mse]:.6f}")
                    
                with col2:
                    st.metric("Melhor Acurácia (Teste)", f"{test_scores_acc[ind_melhor_arvore_acc]:.4f}")
                    st.metric("Alpha correspondente", f"{ccp_alphas_reduzidos[ind_melhor_arvore_acc]:.6f}")

                # Download dos dataframes processados
                st.subheader("💾 Download dos Dados Processados")
                
                csv_treino = dataframe_treino.to_csv(index=False)
                csv_teste = dataframe_teste.to_csv(index=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="Download DataFrame Treino",
                        data=csv_treino,
                        file_name="dataframe_treino.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.download_button(
                        label="Download DataFrame Teste",
                        data=csv_teste,
                        file_name="dataframe_teste.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Erro ao processar os dados: {str(e)}")
        st.info("Verifique se os arquivos estão no formato correto.")

else:
    st.info("⚠️ Por favor, faça upload de todos os arquivos necessários para iniciar a análise.")
    
    st.markdown("""
    ### Arquivos necessários:
    - **features.txt**: Nomes das features
    - **subject_train.txt**: Identificadores dos indivíduos (treino)
    - **X_train.txt**: Dados de treino
    - **y_train.txt**: Labels de treino
    - **subject_test.txt**: Identificadores dos indivíduos (teste)
    - **X_test.txt**: Dados de teste
    - **y_test.txt**: Labels de teste
    """)