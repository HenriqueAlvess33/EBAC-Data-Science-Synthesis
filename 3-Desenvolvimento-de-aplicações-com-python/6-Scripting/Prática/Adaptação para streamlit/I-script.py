import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os

# Configuração da página
st.set_page_config(page_title="Análise SINASC", layout="wide")

# Título da aplicação
st.title("📊 Análise de Dados de Nascimento - SINASC 2019")

# Sidebar para configurações
st.sidebar.header("Configurações")

# Lista de arquivos CSV
lista_de_dataframes = [
    "./input/SINASC_RO_2019_MAR.csv",
    "./input/SINASC_RO_2019_ABR.csv",
    "./input/SINASC_RO_2019_MAI.csv",
    "./input/SINASC_RO_2019_JUN.csv",
    "./input/SINASC_RO_2019_JUL.csv",
    "./input/SINASC_RO_2019_AGO.csv",
    "./input/SINASC_RO_2019_SET.csv",
    "./input/SINASC_RO_2019_OUT.csv",
    "./input/SINASC_RO_2019_NOV.csv",
    "./input/SINASC_RO_2019_DEZ.csv",
]

def plota_pivot_table(dataframe, values, index, funcao, ylabel, xlabel, opcao="nada"):
    """
    Função para criar gráficos a partir de pivot tables
    """
    fig, ax = plt.subplots(figsize=[12, 6])
    
    if opcao == "nada":
        pd.pivot_table(dataframe, values=values, index=index, aggfunc=funcao).plot(ax=ax)
    elif opcao == "unstack":
        pd.pivot_table(dataframe, values=values, index=index, aggfunc=funcao).unstack().plot(ax=ax)
    elif opcao == "sort_values":
        pd.pivot_table(dataframe, values=values, index=index, aggfunc=funcao).sort_values(values).plot(ax=ax)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    
    return fig

# Seleção do mês no sidebar
mes_selecionado = st.sidebar.selectbox(
    "Selecione o mês:",
    options=[df.split('_')[-1].split('.')[0] for df in lista_de_dataframes],
    index=0
)

# Encontrar o arquivo correspondente ao mês selecionado
arquivo_selecionado = next((df for df in lista_de_dataframes if mes_selecionado in df), lista_de_dataframes[0])

# Carregar dados
@st.cache_data
def carregar_dados(arquivo):
    return pd.read_csv(arquivo)

try:
    sinasc = carregar_dados(arquivo_selecionado)
    
    # Informações básicas
    st.sidebar.subheader("Informações do Dataset")
    st.sidebar.write(f"Registros: {len(sinasc):,}")
    st.sidebar.write(f"Período: {sinasc['DTNASC'].min()} a {sinasc['DTNASC'].max()}")
    
    # Mostrar dataframe
    with st.expander("Visualizar Dados Brutos"):
        st.dataframe(sinasc.head(100))
        st.write(f"Shape do dataset: {sinasc.shape}")
    
    # Gráficos
    st.header("📈 Visualizações")
    
    # Layout em colunas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Média de Idade das Mães por Data")
        fig1 = plota_pivot_table(
            sinasc, "IDADEMAE", "DTNASC", "mean", 
            "Idade média das mães", "Data de nascimento"
        )
        st.pyplot(fig1)
        
        st.subheader("Contagem de Nascimentos por Data e Sexo")
        fig3 = plota_pivot_table(
            sinasc, "IDADEMAE", ["DTNASC", "SEXO"], "count",
            "Quantidade de nascimentos", "Data de nascimento", "unstack"
        )
        st.pyplot(fig3)
        
        st.subheader("Peso dos Bebês vs Escolaridade das Mães")
        fig5 = plota_pivot_table(
            sinasc, "PESO", "ESCMAE", "median",
            "Peso médio do bebê (g)", "Escolaridade da mãe", "nada"
        )
        st.pyplot(fig5)
    
    with col2:
        st.subheader("Quantidade de Nascimentos por Data")
        fig2 = plota_pivot_table(
            sinasc, "IDADEMAE", "DTNASC", "count",
            "Quantidade de nascimentos", "Data de nascimento"
        )
        st.pyplot(fig2)
        
        st.subheader("Peso dos Recém-Nascidos por Data e Sexo")
        fig4 = plota_pivot_table(
            sinasc, "PESO", ["DTNASC", "SEXO"], "count",
            "Quantidade de nascimentos", "Data de nascimento", "unstack"
        )
        st.pyplot(fig4)
    
    # APGAR scores
    st.subheader("Índices APGAR vs Tempo de Gestação")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("APGAR1 por Tempo de Gestação")
        fig6 = plota_pivot_table(
            sinasc, "APGAR1", "GESTACAO", "mean",
            "APGAR1 médio", "Semanas de gestação", "sort_values"
        )
        st.pyplot(fig6)
    
    with col4:
        st.write("APGAR5 por Tempo de Gestação")
        fig7 = plota_pivot_table(
            sinasc, "APGAR5", "GESTACAO", "mean",
            "APGAR5 médio", "Semanas de gestação", "sort_values"
        )
        st.pyplot(fig7)
    
    # Estatísticas rápidas
    st.header("📊 Estatísticas Descritivas")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Idade Média das Mães", f"{sinasc['IDADEMAE'].mean():.1f} anos")
    
    with col6:
        st.metric("Peso Médio ao Nascer", f"{sinasc['PESO'].mean():.1f} g")
    
    with col7:
        st.metric("APGAR1 Médio", f"{sinasc['APGAR1'].mean():.1f}")
    
    with col8:
        st.metric("APGAR5 Médio", f"{sinasc['APGAR5'].mean():.1f}")
    
    # Distribuição por sexo
    st.subheader("Distribuição por Sexo")
    distribuicao_sexo = sinasc['SEXO'].value_counts()
    fig_pizza, ax = plt.subplots()
    ax.pie(distribuicao_sexo.values, labels=distribuicao_sexo.index, autopct='%1.1f%%')
    st.pyplot(fig_pizza)

except FileNotFoundError:
    st.error(f"Arquivo {arquivo_selecionado} não encontrado!")
    st.info("Certifique-se de que os arquivos CSV estão na pasta 'input/'")
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")

# Rodapé
st.sidebar.markdown("---")
st.sidebar.info("SINASC 2019 - Sistema de Informações sobre Nascidos Vivos")