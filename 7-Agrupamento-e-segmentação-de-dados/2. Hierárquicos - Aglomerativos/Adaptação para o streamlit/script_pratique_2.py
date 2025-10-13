import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from gower import gower_matrix
from scipy.spatial.distance import squareform
from ucimlrepo import fetch_ucirepo

# Configuração da página
st.set_page_config(page_title="Agrupamento Hierárquico - Comportamento de Navegação", 
                   page_icon="🛒", layout="wide")

# Título da aplicação
st.title("🛒 Análise de Agrupamento Hierárquico - Comportamento de Navegação")
st.markdown("""
Esta aplicação realiza agrupamento hierárquico nas sessões de acesso ao portal considerando 
o comportamento de acesso e informações da data.
""")

# Definir variáveis globais
variaveis_selecionadas = [
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates",
    "PageValues", "SpecialDay", "Month", "OperatingSystems", "TrafficType",
    "VisitorType", "Weekend", "Revenue"
]

variaveis_numericas = ["ProductRelated", "ProductRelated_Duration", "BounceRates", 
                      "ExitRates", "PageValues", "SpecialDay"]

variaveis_categoricas = ["Month", "OperatingSystems", "TrafficType", "VisitorType"]

variaveis_booleanas = ["Weekend", "Revenue"]

# Sidebar para controles
st.sidebar.header("Configurações do Agrupamento")

# Configurações do agrupamento na sidebar
n_clusters = st.sidebar.slider("Número de clusters", min_value=2, max_value=6, value=3)
metodo_linkage = st.sidebar.selectbox("Método de ligação", 
                                    ["complete", "ward", "average", "single"])

# Carregamento dos dados
@st.cache_data
def load_data():
    """Carrega e prepara os dados"""
    try:
        online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468)
        X = online_shoppers_purchasing_intention_dataset.data.features
        y = online_shoppers_purchasing_intention_dataset.data.targets
        df_original = pd.concat([X, y], axis=1)
        return df_original
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

# Carregar dados
with st.spinner('Carregando dados...'):
    df_original = load_data()

if df_original is not None:
    
    df_variaveis_selecionadas = df_original[variaveis_selecionadas]
    
    # Análise Descritiva
    st.header("📊 Análise Descritiva")
    
    # Seleção de visualizações
    analise_option = st.selectbox(
        "Selecione o tipo de análise:",
        ["Visão Geral", "Variáveis Numéricas", "Variáveis Categóricas", "Variáveis Booleanas"]
    )
    
    if analise_option == "Visão Geral":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Primeiras linhas dos dados")
            st.dataframe(df_variaveis_selecionadas.head())
        with col2:
            st.subheader("Informações dos dados")
            st.write(f"**Formato dos dados:** {df_variaveis_selecionadas.shape}")
            st.write(f"**Valores ausentes:** {df_variaveis_selecionadas.isna().sum().sum()}")
            
            # Informações sobre tipos de dados
            st.subheader("Tipos de dados")
            st.write(df_variaveis_selecionadas.dtypes)
    
    elif analise_option == "Variáveis Numéricas":
        st.subheader("Distribuição das Variáveis Numéricas")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, var in enumerate(variaveis_numericas):
            df_variaveis_selecionadas[var].hist(bins=30, ax=axes[i])
            axes[i].set_title(f'Distribuição de {var}')
            axes[i].set_xlabel(var)
            axes[i].set_ylabel('Frequência')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Estatísticas descritivas
        st.subheader("Estatísticas Descritivas - Variáveis Numéricas")
        st.dataframe(df_variaveis_selecionadas[variaveis_numericas].describe())
    
    elif analise_option == "Variáveis Categóricas":
        st.subheader("Frequência das Variáveis Categóricas")
        for var in variaveis_categoricas:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(data=df_variaveis_selecionadas, x=var, 
                             order=df_variaveis_selecionadas[var].value_counts().index, ax=ax)
                ax.set_title(f'Frequência da variável {var}')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            with col2:
                st.write(f"**Valores únicos de {var}:**")
                st.write(df_variaveis_selecionadas[var].value_counts())
    
    elif analise_option == "Variáveis Booleanas":
        st.subheader("Frequência das Variáveis Booleanas")
        col1, col2 = st.columns(2)
        for i, var in enumerate(variaveis_booleanas):
            with col1 if i == 0 else col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df_variaveis_selecionadas, x=var, ax=ax)
                ax.set_title(f'Frequência da variável {var}')
                st.pyplot(fig)
                
                # Porcentagens
                counts = df_variaveis_selecionadas[var].value_counts()
                percentages = df_variaveis_selecionadas[var].value_counts(normalize=True) * 100
                st.write(f"**Distribuição de {var}:**")
                for valor, count, percent in zip(counts.index, counts.values, percentages.values):
                    st.write(f"{valor}: {count} ({percent:.1f}%)")
    
    # Agrupamento Hierárquico
    st.header("🔍 Agrupamento Hierárquico")
    
    if st.button("Executar Agrupamento", type="primary"):
        with st.spinner('Processando agrupamento...'):
            # Preparação dos dados
            df1 = df_variaveis_selecionadas.copy()
            df1 = pd.get_dummies(df1, drop_first=True)
            
            # Identificação de variáveis categóricas
            vars_cat = [False if var in variaveis_selecionadas else True for var in df1.columns]
            
            # Padronização
            df1_std = StandardScaler().fit_transform(df1)
            
            # Matriz de distância Gower
            try:
                distancia_gower = gower_matrix(df1_std, cat_features=vars_cat)
                gdv = squareform(distancia_gower, force='tovector')
                
                # Agrupamento hierárquico
                Z = linkage(gdv, method=metodo_linkage)
                labels = fcluster(Z, n_clusters, criterion='maxclust')
                
                # Adicionar labels ao dataframe
                df_resultado = df_variaveis_selecionadas.copy()
                df_resultado[f'Grupo_{n_clusters}'] = labels
                
                # Resultados
                st.subheader(f"Resultados do Agrupamento - {n_clusters} Clusters")
                
                # Dendrograma
                st.subheader("Dendrograma")
                fig, ax = plt.subplots(figsize=(12, 6))
                dendrogram(
                    Z,
                    truncate_mode='lastp',
                    p=n_clusters,
                    show_leaf_counts=True,
                    ax=ax
                )
                plt.title(f"Dendrograma com {n_clusters} clusters (método: {metodo_linkage})")
                st.pyplot(fig)
                
                # Análise dos clusters
                st.subheader("Análise dos Clusters")
                
                # Tamanho dos clusters
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Distribuição dos clusters:**")
                    cluster_counts = df_resultado[f'Grupo_{n_clusters}'].value_counts().sort_index()
                    st.bar_chart(cluster_counts)
                    
                    # Tabela com contagens
                    st.write("**Contagem por cluster:**")
                    for cluster, count in cluster_counts.items():
                        st.write(f"Grupo {cluster}: {count} observações ({count/len(df_resultado)*100:.1f}%)")
                
                with col2:
                    st.write("**Revenue por cluster:**")
                    revenue_cluster = pd.crosstab(df_resultado[f'Grupo_{n_clusters}'], 
                                                df_resultado['Revenue'], 
                                                normalize='index')
                    st.dataframe(revenue_cluster.style.format("{:.2%}"))
                
                # Estatísticas descritivas por cluster
                st.write("**Estatísticas por cluster (variáveis numéricas):**")
                stats_cluster = df_resultado.groupby(f'Grupo_{n_clusters}')[variaveis_numericas].mean()
                st.dataframe(stats_cluster.style.format("{:.2f}"))
                
                # Visualizações detalhadas
                st.subheader("Visualizações Detalhadas por Cluster")
                
                # Seleção de variável para visualização
                var_visualizacao = st.selectbox("Selecione a variável para visualizar:", 
                                              variaveis_numericas + variaveis_categoricas[:2])
                
                if var_visualizacao in variaveis_numericas:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=df_resultado, x=f'Grupo_{n_clusters}', y=var_visualizacao, ax=ax)
                    ax.set_title(f'Distribuição de {var_visualizacao} por Cluster')
                    st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(data=df_resultado, x=var_visualizacao, hue=f'Grupo_{n_clusters}', ax=ax)
                    ax.set_title(f'Frequência de {var_visualizacao} por Cluster')
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Análise de BounceRates vs Revenue
                st.subheader("📈 Análise: BounceRates vs Revenue")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # BounceRates por cluster
                sns.boxplot(data=df_resultado, x=f'Grupo_{n_clusters}', y='BounceRates', ax=ax1)
                ax1.set_title('BounceRates por Cluster')
                
                # Revenue por cluster
                revenue_data = df_resultado.groupby(f'Grupo_{n_clusters}')['Revenue'].mean().reset_index()
                sns.barplot(data=revenue_data, x=f'Grupo_{n_clusters}', y='Revenue', ax=ax2)
                ax2.set_title('Taxa de Revenue por Cluster')
                ax2.set_ylabel('Proporção de Revenue')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Insights
                st.subheader("💡 Insights")
                
                # Encontrar grupo com maior revenue
                revenue_por_grupo = df_resultado.groupby(f'Grupo_{n_clusters}')['Revenue'].mean()
                grupo_maior_revenue = revenue_por_grupo.idxmax()
                maior_revenue = revenue_por_grupo.max()
                
                # Encontrar grupo com menor bounce rate
                bounce_por_grupo = df_resultado.groupby(f'Grupo_{n_clusters}')['BounceRates'].mean()
                grupo_menor_bounce = bounce_por_grupo.idxmin()
                menor_bounce = bounce_por_grupo.min()
                
                st.write(f"""
                - **Grupo com maior propensão a compra**: Grupo {grupo_maior_revenue} ({maior_revenue:.1%})
                - **Grupo com menor taxa de rejeição**: Grupo {grupo_menor_bounce} ({menor_bounce:.2f} de BounceRate)
                - **Total de observações**: {len(df_resultado):,}
                - **Método de agrupamento**: {metodo_linkage}
                """)
                
                # Download dos resultados
                st.subheader("📥 Download dos Resultados")
                csv = df_resultado.to_csv(index=False)
                st.download_button(
                    label="Baixar dados com clusters",
                    data=csv,
                    file_name=f"dados_clusters_{n_clusters}_{metodo_linkage}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Erro durante o agrupamento: {e}")
                st.info("""
                Possíveis soluções:
                - Verifique se todas as bibliotecas estão instaladas (pip install gower)
                - Tente reduzir o número de clusters
                - Experimente outro método de ligação
                """)

else:
    st.error("Não foi possível carregar os dados. Verifique a conexão com a internet.")

# Informações adicionais
st.sidebar.markdown("---")
st.sidebar.header("Sobre")
st.sidebar.info("""
**Dataset:** Online Shoppers Purchasing Intention
**Fonte:** UCI Machine Learning Repository

**Variáveis utilizadas:**
- Comportamento de navegação
- Informações temporais
- Características do usuário
- Métricas de conversão
""")

st.sidebar.header("Instalação de dependências")
st.sidebar.code("pip install streamlit pandas scikit-learn scipy gower matplotlib seaborn ucimlrepo")