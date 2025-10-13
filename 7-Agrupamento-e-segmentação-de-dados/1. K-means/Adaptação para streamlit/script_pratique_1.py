import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ucimlrepo import fetch_ucirepo

# Configuração da página
st.set_page_config(page_title="Análise de Clusters de Comportamento Online", layout="wide")

# Título da aplicação
st.title("📊 Análise de Clusters de Comportamento Online")
st.markdown("""
Análise de agrupamento de clientes baseada no dataset [Online Shoppers Purchase Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
""")

# Sidebar com controles
st.sidebar.header("Configurações da Análise")

# Configurações na sidebar
max_clusters = st.sidebar.slider(
    "Número máximo de clusters para testar:",
    min_value=2,
    max_value=15,
    value=10,
    help="Define quantas soluções de agrupamento serão testadas"
)

random_state = st.sidebar.number_input(
    "Random State:",
    min_value=0,
    max_value=100,
    value=42,
    help="Semente para reproducibilidade dos resultados"
)

variancia_minima = st.sidebar.slider(
    "Variância mínima explicada pelo PCA:",
    min_value=0.7,
    max_value=0.95,
    value=0.9,
    step=0.05,
    help="Percentual mínimo de variância a ser explicado pelos componentes principais"
)

# Seleção de variáveis para análise
st.sidebar.subheader("Variáveis para Análise")
variaveis_selecionadas = st.sidebar.multiselect(
    "Selecione as variáveis:",
    options=['Administrative', 'Administrative_Duration', 'Informational', 
             'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
             'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'],
    default=['Administrative', 'BounceRates', 'ExitRates'],
    help="Escolha quais variáveis visualizar nos gráficos de análise"
)

# Configurações de visualização
st.sidebar.subheader("Configurações de Visualização")
tema_graficos = st.sidebar.selectbox(
    "Tema dos gráficos:",
    options=['whitegrid', 'darkgrid', 'white', 'dark', 'ticks'],
    index=0,
    help="Estilo dos gráficos do Seaborn"
)

tamanho_fonte = st.sidebar.slider(
    "Tamanho da fonte:",
    min_value=10,
    max_value=18,
    value=12
)

# Aplicar configurações de visualização
sns.set_style(tema_graficos)
plt.rcParams['font.size'] = tamanho_fonte

# Carregamento dos dados
@st.cache_data
def load_data():
    try:
        online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468)
        X = online_shoppers_purchasing_intention_dataset.data.features
        y = online_shoppers_purchasing_intention_dataset.data.targets
        df = pd.concat([X, y], axis=1)
        return df, online_shoppers_purchasing_intention_dataset
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None

# Botão para recarregar dados
if st.sidebar.button("🔄 Recarregar Dados"):
    st.cache_data.clear()

# Informações adicionais na sidebar
st.sidebar.markdown("---")
st.sidebar.header("Sobre a Análise")
st.sidebar.info("""
**Variáveis de escopo:**
- Administrative
- Administrative_Duration  
- Informational
- Informational_Duration
- ProductRelated
- ProductRelated_Duration

**Configurações atuais:**
- Máx. clusters: {}
- Random state: {}
- Variância PCA: {}%
""".format(max_clusters, random_state, int(variancia_minima * 100)))

# Resto do código...
try:
    df, dataset_info = load_data()
    
    if df is not None:
        # Pré-processamento
        numeracao_meses = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
                           'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        tipo_de_visitante = {'Returning_Visitor': 1, 'New_Visitor': 2, 'Other': 3}
        
        df1 = df.copy()
        df1['Month'] = df1['Month'].map(numeracao_meses)
        df1['VisitorType'] = df1['VisitorType'].map(tipo_de_visitante)
        
        # Análise Descritiva
        st.header("📈 Análise Descritiva")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informações do Dataset")
            st.write(f"**Número de registros:** {df.shape[0]}")
            st.write(f"**Número de variáveis:** {df.shape[1]}")
            st.write(f"**Valores ausentes:** {df.isna().sum().sum()}")
            
        with col2:
            st.subheader("Distribuição de Revenue")
            revenue_counts = df['Revenue'].value_counts()
            fig, ax = plt.subplots()
            revenue_counts.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral'])
            ax.set_title('Distribuição de Revenue')
            ax.set_xlabel('Revenue')
            ax.set_ylabel('Frequência')
            st.pyplot(fig)
        
        # PCA Analysis
        st.header("🔍 Análise de Componentes Principais (PCA)")
        
        # Normalização
        df1_padrao = StandardScaler().fit_transform(df1)
        prcomp = PCA(n_components=18).fit(df1_padrao)
        df_pca = prcomp.transform(df1_padrao)
        df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(18)])
        
        # Scree Plot
        st.subheader("Scree Plot - Variância Explicada")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        num_componentes = np.arange(prcomp.n_components_) + 1
        
        # Variância total
        ax1.plot(num_componentes, prcomp.explained_variance_, 'o-', linewidth=2, color='blue')
        ax1.set_title('Variância Total por Componente')
        ax1.set_xlabel('Número de Componentes')
        ax1.set_ylabel('Variância Explicada')
        
        # Variância acumulada
        ax2.plot(num_componentes, prcomp.explained_variance_.cumsum(), 'o-', linewidth=2, color='blue')
        ax2.set_title('Variância Acumulada')
        ax2.set_xlabel('Número de Componentes')
        ax2.set_ylabel('Variância Acumulada')
        
        # Variância percentual
        ax3.plot(num_componentes, prcomp.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
        ax3.set_title('Variância Percentual por Componente')
        ax3.set_xlabel('Número de Componentes')
        ax3.set_ylabel('Variância Explicada (%)')
        
        # Variância percentual acumulada
        ax4.plot(num_componentes, prcomp.explained_variance_ratio_.cumsum(), 'o-', linewidth=2, color='blue')
        ax4.set_title('Variância Percentual Acumulada')
        ax4.set_xlabel('Número de Componentes')
        ax4.set_ylabel('Variância Acumulada (%)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Determinação do número de clusters
        st.header("🎯 Determinação do Número de Clusters")
        
        # Cálculo do silhouette score para dados originais
        silhuette_scores = []
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            kmeans.fit(df1)
            silhuette_scores.append(silhouette_score(df1, kmeans.labels_))
        
        # Plot silhouette scores
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(2, max_clusters + 1), silhuette_scores, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Número de Clusters')
        ax.set_ylabel('Score de Silhouette')
        ax.set_title('Score de Silhouette vs Número de Clusters')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Seleção do melhor número de clusters
        best_n_clusters = np.argmax(silhuette_scores) + 2
        st.info(f"**Número recomendado de clusters:** {best_n_clusters} (Silhouette score: {silhuette_scores[best_n_clusters-2]:.3f})")
        
        # Aplicação do KMeans com o melhor número de clusters
        kmeans_final = KMeans(n_clusters=best_n_clusters, random_state=random_state)
        kmeans_final.fit(df1)
        df_agrupamento = df1.copy()
        nomes_grupos = [f'Grupo_{i}' for i in range(best_n_clusters)]
        df_agrupamento['Cluster'] = pd.Categorical.from_codes(kmeans_final.labels_, categories=nomes_grupos)
        
        # Análise dos Clusters
        st.header("👥 Análise dos Clusters")
        
        # Proporção dos clusters
        st.subheader("Distribuição dos Clusters")
        fig, ax = plt.subplots(figsize=(10, 6))
        proporcao = df_agrupamento['Cluster'].value_counts(normalize=True) * 100
        bars = proporcao.plot(kind='bar', color='lightblue', ax=ax)
        
        for i, valor in enumerate(proporcao):
            ax.text(i, valor + 0.5, f'{valor:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax.set_title(f'Proporção dos Clusters ({best_n_clusters} grupos)')
        ax.set_xlabel('Clusters')
        ax.set_ylabel('Proporção (%)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Análise por variáveis selecionadas
        if variaveis_selecionadas:
            st.subheader("Análise por Variáveis Selecionadas")
            
            for var in variaveis_selecionadas:
                if var in df_agrupamento.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=df_agrupamento, x='Cluster', y=var, hue='Revenue', 
                               errorbar=None, palette='coolwarm', ax=ax)
                    ax.set_title(f'Média de {var} por Cluster e Revenue')
                    ax.set_ylabel(f'Média de {var}')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        
        # Revenue por cluster
        st.subheader("Revenue por Cluster")
        fig, ax = plt.subplots(figsize=(10, 6))
        revenue_by_cluster = df_agrupamento.groupby(['Cluster', 'Revenue']).size().unstack()
        revenue_by_cluster.plot(kind='bar', ax=ax, color=['lightcoral', 'lightgreen'])
        ax.set_title('Distribuição de Revenue por Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Número de Usuários')
        ax.legend(title='Revenue', loc='upper right')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Insights
        st.header("💡 Insights e Conclusões")
        
        # Calcular métricas por cluster
        cluster_stats = df_agrupamento.groupby('Cluster').agg({
            'Administrative': 'mean',
            'Informational': 'mean',
            'ProductRelated': 'mean',
            'BounceRates': 'mean',
            'ExitRates': 'mean',
            'Revenue': 'mean'
        }).round(3)
        
        st.subheader("Estatísticas por Cluster")
        st.dataframe(cluster_stats)
        
        st.markdown("""
        **Principais observações:**
        - Usuários que acessam mais páginas administrativas e informativas tendem a ter maior taxa de conversão
        - Altos valores de Bounce Rate e Exit Rate indicam menor engajamento e propensão à compra
        - O comportamento de navegação está claramente refletido nos grupos formados
        """)
    
except Exception as e:
    st.error(f"Erro na análise: {e}")