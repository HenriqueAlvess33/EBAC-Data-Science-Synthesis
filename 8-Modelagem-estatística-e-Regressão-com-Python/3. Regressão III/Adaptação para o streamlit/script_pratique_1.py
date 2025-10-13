import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score, mean_squared_error

# Configuração da página
st.set_page_config(page_title="Análise de Regressão", layout="wide")
st.title("📊 Regressão III - Análise de Previsão de Renda")

# Sidebar para upload de arquivos
st.sidebar.header("Upload de Dados")
uploaded_file = st.sidebar.file_uploader(
    "Faça upload do arquivo CSV", 
    type=['csv'],
    help="Upload do arquivo de dados para análise"
)

if uploaded_file is not None:
    try:
        # Carregar dados
        df = pd.read_csv(uploaded_file, index_col=0)
        df = df.reset_index()
        
        st.success("✅ Arquivo carregado com sucesso!")
        
        # Mostrar informações básicas do dataset
        st.header("📋 Visualização dos Dados")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Linhas", df.shape[0])
        with col2:
            st.metric("Total de Colunas", df.shape[1])
        with col3:
            st.metric("Valores Missing", df.isna().sum().sum())
        
        # Mostrar primeiras linhas
        st.subheader("Primeiras linhas do dataset")
        st.dataframe(df.head())
        
        # Tratamento de valores missing
        st.header("🔧 Tratamento de Dados")
        
        if st.checkbox("Mostrar valores missing por coluna"):
            st.write("Valores missing antes do tratamento:")
            st.dataframe(df.isna().sum())
        
        # Substituir valores missing pela média
        if 'tempo_emprego' in df.columns:
            df_original = df.copy()
            df.loc[df['tempo_emprego'].isna(), 'tempo_emprego'] = df['tempo_emprego'].mean()
            st.success("Valores missing em 'tempo_emprego' substituídos pela média")
        
        # Divisão da base de dados
        st.header("📊 Divisão da Base de Dados")
        
        if 'data_ref' in df.columns:
            # Base de teste (últimos 3 meses)
            df_last_three_months = df.loc[df['data_ref'] >= '2016-01-01']
            df_train = df.loc[df['data_ref'] < '2016-01-01']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Base de Treino", f"{df_train.shape[0]:,} linhas")
            with col2:
                st.metric("Base de Teste", f"{df_last_three_months.shape[0]:,} linhas")
        else:
            st.warning("Coluna 'data_ref' não encontrada. Usando dataset completo para análise.")
            df_train = df.copy()
            df_last_three_months = pd.DataFrame()
        
        # Análise de Perfil
        st.header("📈 Análise de Perfil")
        
        if 'tempo_emprego' in df_train.columns and 'renda' in df_train.columns:
            # Criar categorias para tempo_emprego
            grupos = pd.qcut(df_train.tempo_emprego, 20, duplicates='drop')
            
            # Preparar dados para análise
            variaveis = [col for col in df_train.columns if col not in ['data_ref', 'index']]
            tab = df_train[variaveis].copy()
            tab['log_renda'] = tab['renda'].apply(lambda x: 0 if x <= 0 else np.log(x))
            tab = tab.select_dtypes(include=['float64', 'int64'])
            tab = tab.groupby(grupos).mean()
            
            # Gráficos
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Renda vs Tempo de Emprego")
                fig, ax = plt.subplots()
                sns.scatterplot(data=tab, x='tempo_emprego', y='renda', ax=ax)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Log-Renda vs Tempo de Emprego")
                fig, ax = plt.subplots()
                sns.scatterplot(data=tab, x='tempo_emprego', y='log_renda', ax=ax)
                st.pyplot(fig)
            
            st.info("""
            **Observação:** A transformação logarítmica melhora a simetria da relação entre 
            'log_renda' e 'tempo_emprego', revelando um padrão mais claro comparado à relação original.
            """)
            
            # Modelagem
            st.header("🤖 Modelagem e Linearização")
            
            # Primeiro modelo (simples)
            st.subheader("1. Modelo Linear Simples")
            res_simple = smf.ols('log_renda ~ tempo_emprego', data=tab).fit()
            y_pred_simple = res_simple.fittedvalues
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Ajustado", f"{res_simple.rsquared_adj:.2%}")
            with col2:
                st.metric("R²", f"{res_simple.rsquared:.2%}")
            
            # Modelo com segmentação
            st.subheader("2. Modelo com Segmentação")
            tab_viz = tab.copy()
            C1 = 4
            tab_viz['X1'] = tab_viz['tempo_emprego']
            tab_viz['X1_1'] = (tab_viz['X1'] <= C1) * tab_viz['X1'] + (tab_viz['X1']>C1) * C1
            tab_viz['X1_2'] = (tab_viz['X1'] > C1)* (tab_viz['X1'] - C1)
            
            res_segmentada = smf.ols('log_renda ~ X1_1 + X1_2', data=tab_viz).fit()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Ajustado", f"{res_segmentada.rsquared_adj:.2%}")
            with col2:
                st.metric("R²", f"{res_segmentada.rsquared:.2%}")
            
            # Modelo LOWESS
            st.subheader("3. Modelo LOWESS")
            lowess = sm.nonparametric.lowess(tab_viz['log_renda'], tab_viz['X1'], frac=2/3)
            lowess_y = lowess[:, 1]
            lowess_x = lowess[:, 0]
            
            f = interp1d(lowess_x, lowess_y, bounds_error=False)
            tab_viz['X1_lowess'] = f(tab_viz.X1)
            
            res_lowess = smf.ols('log_renda ~ X1_lowess', data=tab_viz).fit()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Ajustado", f"{res_lowess.rsquared_adj:.2%}")
            with col2:
                st.metric("R²", f"{res_lowess.rsquared:.2%}")
            
            # Comparação de Modelos
            st.header("📊 Comparação dos Modelos")
            
            comparison_data = {
                'Modelo': ['Linear Simples', 'Segmentação', 'LOWESS'],
                'R²': [
                    f"{res_simple.rsquared:.2%}",
                    f"{res_segmentada.rsquared:.2%}", 
                    f"{res_lowess.rsquared:.2%}"
                ],
                'R² Ajustado': [
                    f"{res_simple.rsquared_adj:.2%}",
                    f"{res_segmentada.rsquared_adj:.2%}",
                    f"{res_lowess.rsquared_adj:.2%}"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Avaliação na base de teste
            if not df_last_three_months.empty:
                st.header("🧪 Avaliação na Base de Teste")
                
                tab_teste = df_last_three_months[variaveis].copy()
                tab_teste['log_renda'] = tab_teste['renda'].apply(lambda x: 0 if x <= 0 else np.log(x))
                tab_teste = tab_teste.select_dtypes(include=['float64', 'int64'])
                
                # Preparar dados para predição
                tab_teste['X1'] = tab_teste['tempo_emprego']
                tab_teste['X1_1'] = (tab_teste['X1'] <= C1) * tab_teste['X1'] + (tab_teste['X1']>C1) * C1
                tab_teste['X1_2'] = (tab_teste['X1'] > C1)* (tab_teste['X1'] - C1)
                
                y_pred_test = res_segmentada.predict(tab_teste[['X1_1', 'X1_2']])
                r2_test = r2_score(tab_teste['log_renda'], y_pred_test)
                
                st.metric("R² na Base de Teste", f"{r2_test:.2%}")
                
        else:
            st.error("Colunas 'tempo_emprego' ou 'renda' não encontradas no dataset.")
    
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {str(e)}")
else:
    st.info("👆 Faça upload de um arquivo CSV para iniciar a análise.")
    
    # Exemplo de estrutura esperada
    st.subheader("Estrutura esperada do arquivo:")
    st.code("""
    Colunas esperadas:
    - data_ref: data de referência
    - tempo_emprego: tempo no emprego atual
    - renda: renda do cliente
    - outras variáveis demográficas e socioeconômicas
    """)

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido para análise de regressão com Streamlit")