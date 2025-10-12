import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from io import StringIO

# Configuração da página
st.set_page_config(page_title="Análise de Doença Cardíaca", layout="wide")

# Título da aplicação
st.title("📊 Análise de Dados de Doença Cardíaca")
st.markdown("---")

# Área para upload de arquivos
st.header("📁 Upload de Dados")
uploaded_file = st.file_uploader(
    "Faça upload do arquivo CSV com os dados", 
    type=['csv'],
    help="O arquivo deve conter as colunas: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num"
)

# Opção para usar dados padrão ou upload
use_default_data = st.checkbox("Usar dados de exemplo (Cleveland Heart Disease)", value=True)

df = None

if uploaded_file is not None:
    # Carregar dados do arquivo upload
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Arquivo carregado com sucesso!")
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {e}")
elif use_default_data:
    # Carregar dados padrão
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        df = pd.read_csv(url, 
                        names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'])
        st.info("📋 Dados de exemplo carregados (Cleveland Heart Disease)")
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados padrão: {e}")

if df is not None:
    # Mostrar informações básicas do dataset
    st.header("📈 Visualização dos Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Primeiras linhas do dataset")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Informações do dataset")
        st.write(f"**Formato:** {df.shape[0]} linhas × {df.shape[1]} colunas")
        st.write("**Colunas:**", list(df.columns))
    
    # Processamento dos dados
    st.header("🔧 Processamento dos Dados")
    
    # Criar flag_doente
    df['flag_doente'] = (df['num'] != 0).astype('int64')
    
    # Análise por sexo
    st.subheader("📋 Análise por Sexo")
    
    tab0 = (
        pd.crosstab(df["sex"], df["flag_doente"], margins=True)
        .rename(columns={'All': 'Total'}, index={0.0:'F', 1.0:'M','All':'Total'})
        .rename_axis(index="Sexo", columns="Diagnosticados")
    )
    
    tab0['mean_diagnosticos'] = round(np.mean(tab0.loc[:, tab0.columns != 'Total'], axis=1), 2)
    tab0['Odds'] = tab0[1] / tab0[0]
    tab0['Odd_ratio vs Total'] = tab0['Odds'] / tab0.loc['Total', 'Odds']
    tab0 = tab0.reindex(columns=[0, 1, 'mean_diagnosticos', 'Odds', 'Odd_ratio vs Total', 'Total'])
    
    st.dataframe(tab0)
    
    # Análise por idade
    st.subheader("📊 Análise por Grupos de Idade")
    
    df['cat_age'] = pd.qcut(df['age'], 5)
    
    tab1 = (
        pd.crosstab(df["cat_age"], df["flag_doente"], margins=True)
        .rename(
            columns={"All": "Total"},
            index={
                "All": "Total",
                pd.Interval(28.999, 45.0): "Dos 28 aos 45 anos",
                pd.Interval(45.0, 53.0): "Dos 45 aos 53 anos",
                pd.Interval(53.0, 58.0): "Dos 53 aos 58 anos",
                pd.Interval(58.0, 62.0): "Dos 58 aos 62 anos",
                pd.Interval(62.0, 77.0): "Dos 66 aos 77 anos",
            },
        )
        .rename_axis(index="Grupos de idade", columns="Diagnosticados")
    )
    tab1.index = tab1.index.astype(str)
    
    tab1['mean_diagnosticos'] = np.mean(tab1.loc[:, tab1.columns != 'Total'], axis=1)
    tab1['Odds'] = tab1[1] / tab1[0]
    tab1['Odd_ratio vs Total'] = tab1['Odds'] / tab1.loc['Total', 'Odds']
    tab1 = tab1.reindex(columns=[0, 1, 'mean_diagnosticos', 'Odds', 'Odd_ratio vs Total', 'Total'])
    
    st.dataframe(tab1)
    
    # Visualizações
    st.header("📈 Visualizações Gráficas")
    
    # Configurar o estilo dos gráficos
    sns.set_style("whitegrid")
    
    # Gráfico 1: Distribuição por sexo
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição por Sexo")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, x='sex', hue='flag_doente', ax=ax1)
        ax1.set_xticks(range(2))
        ax1.set_xticklabels(['Feminino', 'Masculino'], rotation=45)
        ax1.set_ylabel('Contagem')
        ax1.set_xlabel('Sexo do paciente')
        ax1.legend(title='Diagnósticos', labels=['Não Doente', 'Doente'])
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Distribuição por Idade")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, hue='flag_doente', x='cat_age', ax=ax2)
        ax2.set_xticks(range(5))
        ax2.set_xticklabels([
            'Dos 28 aos 45 anos', 
            'Dos 45 aos 53 anos', 
            'Dos 53 aos 58 anos',
            'Dos 58 aos 62 anos', 
            'Dos 66 aos 77 anos'
        ], rotation=45)
        ax2.set_ylabel('Contagem')
        ax2.set_xlabel('Categorias de Idade')
        ax2.legend(title='Diagnósticos', labels=['Não Doente', 'Doente'])
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Gráfico 3: Odds Ratio
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Odds Ratio - Sexo")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        tab0['Odd_ratio vs Total'].plot.bar(ax=ax3)
        ax3.set_ylabel('Odds Ratio vs Total')
        ax3.set_xlabel('Sexo')
        plt.tight_layout()
        st.pyplot(fig3)
    
    with col4:
        st.subheader("Odds Ratio - Idade")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        tab1['Odd_ratio vs Total'].plot.bar(ax=ax4)
        ax4.set_ylabel('Odds Ratio vs Total')
        ax4.set_xlabel('Grupos de Idade')
        plt.tight_layout()
        st.pyplot(fig4)
    
    # Análise detalhada por idade
    st.subheader("📊 Análise Detalhada por Idade")
    
    tab1_reset = tab1.reset_index()
    colunas_para_melt = [0, 1]
    colunas_para_manter = ['Grupos de idade', "mean_diagnosticos", "Odds", "Odd_ratio vs Total", "Total"]
    
    tab1_melted = tab1_reset.melt(
        id_vars=["Grupos de idade"],
        var_name="Diagnosticos",
        value_vars=colunas_para_melt,
        value_name="Estatísticas",
    )
    
    tab1_visual = tab1_melted.merge(tab1_reset[colunas_para_manter], on='Grupos de idade')
    tab1_visual = tab1_visual.rename(columns={'Diagnosticos': 'flag_doente'})
    
    # Boxplot
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=tab1_visual, x="Grupos de idade", y="Odds", hue="flag_doente", ax=ax5)
    ax5.set_xticklabels([
        "Dos 28 aos 45 anos",
        "Dos 45 aos 53 anos",
        "Dos 53 aos 58 anos",
        "Dos 58 aos 62 anos",
        "Dos 66 aos 77 anos",
        "Total",
    ], rotation=45)
    ax5.set_ylabel('Odds')
    ax5.set_xlabel('Grupos de Idade')
    ax5.legend(title='Diagnóstico', labels=['Não Doente', 'Doente'])
    plt.tight_layout()
    st.pyplot(fig5)
    
    # Informações adicionais
    st.header("ℹ️ Informações Adicionais")
    
    with st.expander("Sobre os dados"):
        st.markdown("""
        - **age**: Idade em anos
        - **sex**: Sexo (1 = masculino; 0 = feminino)
        - **cp**: Tipo de dor no peito
        - **trestbps**: Pressão arterial em repouso
        - **chol**: Colesterol sérico em mg/dl
        - **fbs**: Açúcar no sangue em jejum > 120 mg/dl
        - **restecg**: Resultados eletrocardiográficos em repouso
        - **thalach**: Frequência cardíaca máxima alcançada
        - **exang**: Angina induzida por exercício
        - **oldpeak**: Depressão do ST induzida por exercício
        - **slope**: Inclinação do segmento ST de pico do exercício
        - **ca**: Número de vasos principais coloridos por fluoroscopia
        - **thal**: Talassemia
        - **num**: Diagnóstico de doença cardíaca (0 = não; 1,2,3,4 = sim)
        """)
    
    # Download dos dados processados
    st.header("💾 Download dos Dados Processados")
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Baixar dados processados como CSV",
        data=csv,
        file_name="dados_cardiacos_processados.csv",
        mime="text/csv"
    )

else:
    st.warning("⚠️ Por favor, faça upload de um arquivo CSV ou use os dados de exemplo para iniciar a análise.")