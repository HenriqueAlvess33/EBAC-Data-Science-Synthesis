import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import io

# Configuração da página
st.set_page_config(page_title="Análise de Árvore de Decisão", layout="wide")

st.title("📊 Análise de Classificação com Árvore de Decisão")
st.markdown("---")

# Área de upload de arquivos
st.sidebar.header("📁 Upload dos Arquivos")

uploaded_file_treino = st.sidebar.file_uploader(
    "Carregar DataFrame de Treino (CSV)", 
    type=['csv'],
    key='treino'
)

uploaded_file_teste = st.sidebar.file_uploader(
    "Carregar DataFrame de Teste (CSV)", 
    type=['csv'],
    key='teste'
)

# Verifica se os arquivos foram carregados
if uploaded_file_treino is not None and uploaded_file_teste is not None:
    try:
        # Carrega os dados
        dataframe_treino = pd.read_csv(uploaded_file_treino)
        dataframe_teste = pd.read_csv(uploaded_file_teste)
        
        st.success("✅ Arquivos carregados com sucesso!")
        
        # Mostra informações básicas dos dados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("DataFrame de Treino")
            st.write(f"**Formato:** {dataframe_treino.shape}")
            st.dataframe(dataframe_treino.head())
            
        with col2:
            st.subheader("DataFrame de Teste")
            st.write(f"**Formato:** {dataframe_teste.shape}")
            st.dataframe(dataframe_teste.head())
        
        # Primeira Árvore de Decisão
        st.markdown("---")
        st.header("🌳 Primeira Árvore de Decisão")
        
        # Define filtro de colunas
        filtro_de_colunas = ['53 tGravityAcc-min()-X', '272 fBodyAcc-mad()-X', '560 angle(Y,gravityMean)']
        
        # Verifica se as colunas existem nos dataframes
        colunas_existentes_treino = [col for col in filtro_de_colunas if col in dataframe_treino.columns]
        colunas_existentes_teste = [col for col in filtro_de_colunas if col in dataframe_teste.columns]
        
        if len(colunas_existentes_treino) == len(filtro_de_colunas) and len(colunas_existentes_teste) == len(filtro_de_colunas):
            # Prepara os dados
            X_train = dataframe_treino.drop(columns=['Atividades'])[filtro_de_colunas]
            y_train = dataframe_treino['Atividades']
            X_test = dataframe_teste.drop(columns=['Atividades'])[filtro_de_colunas]
            y_test = dataframe_teste['Atividades']
            
            # Treinamento da primeira árvore
            with st.spinner('Treinando primeira árvore de decisão...'):
                caminho = DecisionTreeClassifier(random_state=100).cost_complexity_pruning_path(X_train, y_train)
                ccp_alphas, impurities = caminho.ccp_alphas, caminho.impurities
                ccp_alphas = np.unique(ccp_alphas[ccp_alphas >= 0])
                
                # Grid Search
                clf0 = DecisionTreeClassifier()
                grid_params0 = {'ccp_alpha': ccp_alphas[::10], 'min_samples_leaf': [20]}
                grid0 = GridSearchCV(estimator=clf0, param_grid=grid_params0, cv=10, verbose=0)
                grid0.fit(X_train, y_train)
                
                # Melhor modelo
                melhor_arvore0 = DecisionTreeClassifier(
                    ccp_alpha=grid0.best_params_['ccp_alpha'], 
                    min_samples_leaf=20
                )
                melhor_arvore0.fit(X_train, y_train)
            
            # Resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Resultados da Primeira Árvore")
                st.write(f"**Melhor ccp_alpha:** {grid0.best_params_['ccp_alpha']:.6f}")
                st.write(f"**Melhor score:** {grid0.best_score_:.4f}")
                
                # Matriz de Confusão
                cm = confusion_matrix(y_test, melhor_arvore0.predict(X_test))
                fig, ax = plt.subplots(figsize=(8, 6))
                disp0 = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp0.plot(ax=ax)
                st.pyplot(fig)
            
            with col2:
                # Análise de Erros
                st.subheader("Análise de Erros")
                dataframe_erros = (X_test.copy()
                                  .assign(y_pred=melhor_arvore0.predict(X_test))
                                  .assign(y_true=y_test.values)
                                  .assign(ERROS=lambda df: df['y_pred'] != df['y_true']))
                
                erros_por_classe = dataframe_erros.pivot_table(
                    values='ERROS', 
                    index=dataframe_erros['y_true'], 
                    aggfunc='sum'
                ).sort_values(ascending=False, by='ERROS')
                
                st.dataframe(erros_por_classe)
        
        else:
            st.warning("⚠️ Algumas colunas do filtro não foram encontradas nos dataframes.")
            st.write("Colunas esperadas:", filtro_de_colunas)
            st.write("Colunas encontradas no treino:", colunas_existentes_treino)
            st.write("Colunas encontradas no teste:", colunas_existentes_teste)
        
        # Segunda Árvore - Análise de Erros
        st.markdown("---")
        st.header("🔍 Segunda Árvore - Análise de Erros da Atividade 3")
        
        # Prepara dados para segunda árvore
        dataframe_treino['Atividade_maior_erro_flag'] = [1 if x == 3 else 0 for x in dataframe_treino['Atividades']]
        dataframe_teste['Atividade_maior_erro_flag'] = [1 if x == 3 else 0 for x in dataframe_teste['Atividades']]
        
        X_train_2 = dataframe_treino.drop(columns=['Atividades', 'Atividade_maior_erro_flag'])
        y_train_2 = dataframe_treino['Atividade_maior_erro_flag']
        X_test_2 = dataframe_teste.drop(columns=['Atividades', 'Atividade_maior_erro_flag'])
        y_test_2 = dataframe_teste['Atividade_maior_erro_flag']
        
        # Treinamento da segunda árvore
        with st.spinner('Treinando segunda árvore de decisão...'):
            caminho = DecisionTreeClassifier(
                random_state=99785236, 
                min_samples_leaf=20, 
                max_depth=4
            ).cost_complexity_pruning_path(X_train_2, y_train_2)
            
            ccp_alphas, impurities = caminho.ccp_alphas, caminho.impurities
            ccp_alphas = np.unique(ccp_alphas[ccp_alphas >= 0])
            
            clf1 = DecisionTreeClassifier(random_state=651498)
            grid_params1 = {'min_samples_leaf': [20], 'max_depth': [4], 'ccp_alpha': ccp_alphas}
            grid1 = GridSearchCV(estimator=clf1, param_grid=grid_params1, cv=10, verbose=0)
            grid1.fit(X_train_2, y_train_2)
            
            melhor_arvore1 = DecisionTreeClassifier(
                max_depth=4, 
                min_samples_leaf=20, 
                random_state=651498, 
                ccp_alpha=grid1.best_params_['ccp_alpha']
            )
            melhor_arvore1.fit(X_train_2, y_train_2)
        
        # Resultados da segunda árvore
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Resultados da Segunda Árvore")
            st.write(f"**Melhores parâmetros:** {grid1.best_params_}")
            
            # Matriz de Confusão
            cm = confusion_matrix(y_test_2, melhor_arvore1.predict(X_test_2))
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Importância das Variáveis")
            # Top 3 variáveis mais importantes
            importancia_df = (
                pd.DataFrame({
                    "Importância": melhor_arvore1.feature_importances_,
                    "Variável": X_train_2.columns,
                })
                .sort_values(ascending=False, by="Importância")
                .head(3)
            )
            st.dataframe(importancia_df)
            
            # Análise de erros
            df_erros = (X_test_2.copy()
                       .assign(y_pred=melhor_arvore1.predict(X_test_2))
                       .assign(y_true=y_test_2.values)
                       .assign(ERROS=lambda df: df['y_pred'] != df['y_true']))
            
            st.subheader("Erros por Classe")
            erros_classe_2 = df_erros.pivot_table(
                values='ERROS', 
                index=df_erros['y_true'], 
                aggfunc='sum'
            ).sort_values(ascending=False, by='ERROS')
            st.dataframe(erros_classe_2)
        
        # Terceira Árvore
        st.markdown("---")
        st.header("🌲 Terceira Árvore - Variáveis Mais Importantes")
        
        # Usa as 3 variáveis mais importantes da segunda árvore
        top_3_variaveis = importancia_df['Variável'].head(3).tolist()
        
        if len(top_3_variaveis) == 3:
            X_train_3 = X_train_2[top_3_variaveis]
            y_train_3 = dataframe_treino['Atividades']
            X_test_3 = X_test_2[top_3_variaveis]
            y_test_3 = dataframe_teste['Atividades']
            
            # Treinamento da terceira árvore
            with st.spinner('Treinando terceira árvore de decisão...'):
                clf2 = DecisionTreeClassifier(random_state=26583310)
                caminho2 = DecisionTreeClassifier(random_state=26583310).cost_complexity_pruning_path(X_train_3, y_train_3)
                ccp_alphas, impurities = caminho2.ccp_alphas, caminho2.impurities
                ccp_alphas = np.unique(ccp_alphas[ccp_alphas >= 0])
                
                grid_params2 = {'ccp_alpha': ccp_alphas}
                grid2 = GridSearchCV(estimator=clf2, param_grid=grid_params2, cv=15, verbose=0)
                grid2.fit(X_train_3, y_train_3)
            
            # Comparação final
            st.subheader("📊 Comparação dos Modelos")
            comparacao = pd.DataFrame({
                'Modelo': ['Primeira Árvore', 'Terceira Árvore'],
                'Melhor Score': [grid0.best_score_, grid2.best_score_]
            })
            st.dataframe(comparacao)
            
            # Gráfico de comparação
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(comparacao['Modelo'], comparacao['Melhor Score'], color=['skyblue', 'lightcoral'])
            ax.set_ylabel('Score')
            ax.set_title('Comparação do Desempenho dos Modelos')
            ax.set_ylim(0, 1)
            for i, v in enumerate(comparacao['Melhor Score']):
                ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"❌ Erro ao processar os arquivos: {str(e)}")
        st.info("Verifique se os arquivos possuem a estrutura esperada (coluna 'Atividades')")

else:
    st.info("👆 Por favor, faça o upload dos arquivos CSV de treino e teste para iniciar a análise.")

# Instruções de uso
with st.sidebar.expander("ℹ️ Instruções"):
    st.markdown("""
    **Como usar:**
    1. Faça upload dos arquivos CSV de treino e teste
    2. Os arquivos devem conter uma coluna chamada 'Atividades'
    3. Aguarde o processamento dos modelos
    4. Analise os resultados nas diferentes seções
    
    **Estrutura esperada:**
    - Coluna alvo: 'Atividades'
    - Colunas de características numéricas
    """)
