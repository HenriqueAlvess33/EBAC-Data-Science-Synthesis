import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc

# Configuração da página
st.set_page_config(page_title="Agrupamento Hierárquico - Pinguins", layout="wide")
st.title("Métodos hierárquicos de agrupamento - Análise de Pinguins")

# Carregar dados
@st.cache_data
def load_data():
    return sns.load_dataset("penguins")

penguins = load_data()

# Sidebar para configurações
st.sidebar.header("Configurações do Agrupamento")
n_clusters = st.sidebar.slider("Número de clusters", min_value=2, max_value=6, value=3)
linkage_method = st.sidebar.selectbox(
    "Método de ligação", 
    ["complete", "ward", "average", "single"]
)

# Processamento dos dados
st.header("1) Pré-processamento dos dados")

# Selecionar variáveis quantitativas
penguins_only_numeric = penguins.select_dtypes(include=['float64', 'int64'])

# Remover valores faltantes
penguins_clean = penguins.dropna(subset=penguins_only_numeric.columns)
penguins_only_numeric_clean = penguins_only_numeric.dropna()

st.write(f"**Dados originais:** {len(penguins)} linhas")
st.write(f"**Dados após limpeza:** {len(penguins_clean)} linhas")

# Padronizar os dados
padronizador = StandardScaler()
penguins_only_numeric_std = pd.DataFrame(
    padronizador.fit_transform(penguins_only_numeric_clean), 
    columns=penguins_only_numeric_clean.columns
)

# Agrupamento hierárquico
clus = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage=linkage_method,
    distance_threshold=None,
)

clus.fit(penguins_only_numeric_std)
penguins_clean = penguins_clean.copy()
penguins_clean[f'Grupos_{n_clusters}'] = clus.labels_

# Visualizações
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dendrograma")
    fig, ax = plt.subplots(figsize=(10, 6))
    dend = shc.dendrogram(
        shc.linkage(penguins_only_numeric_std, method=linkage_method),
        ax=ax
    )
    ax.set_title(f'Dendrograma - Método {linkage_method.capitalize()}')
    st.pyplot(fig)

with col2:
    st.subheader("Distribuição por Espécie")
    fig, ax = plt.subplots(figsize=(10, 6))
    cross_tab = pd.crosstab(
        penguins_clean['species'], 
        penguins_clean[f'Grupos_{n_clusters}'],
        rownames=['Espécies'],
        colnames=[f'{n_clusters} Grupos']
    )
    st.write(cross_tab)
    
    # Gráfico de barras
    cross_tab.plot(kind='bar', ax=ax)
    ax.set_title(f'Distribuição das Espécies por Grupo ({n_clusters} grupos)')
    ax.legend(title='Grupos')
    ax.set_ylabel('Contagem')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Gráficos de dispersão
st.header("2) Visualizações dos Grupos")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Comprimento vs Profundidade do Bico")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=penguins_clean, 
        x='bill_length_mm', 
        y='bill_depth_mm', 
        hue=f'Grupos_{n_clusters}', 
        style='species', 
        palette='deep',
        ax=ax
    )
    ax.set_title(f'Agrupamento Hierárquico - {n_clusters} Grupos')
    st.pyplot(fig)

with col4:
    st.subheader("Comprimento da Nadadeira vs Massa Corporal")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=penguins_clean, 
        x='flipper_length_mm', 
        y='body_mass_g', 
        hue=f'Grupos_{n_clusters}', 
        style='species', 
        palette='deep',
        ax=ax
    )
    ax.set_title(f'Agrupamento Hierárquico - {n_clusters} Grupos')
    st.pyplot(fig)

# Análises adicionais
st.header("3) Análises Detalhadas")

col5, col6 = st.columns(2)

with col5:
    st.subheader("Distribuição por Ilha")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=penguins_clean, 
        x=f'Grupos_{n_clusters}', 
        hue='island', 
        palette='deep',
        ax=ax
    )
    ax.set_title(f'Distribuição dos Grupos ({n_clusters}) por Ilha')
    ax.set_xlabel('Grupo')
    ax.set_ylabel('Contagem')
    st.pyplot(fig)

with col6:
    st.subheader("Distribuição por Sexo")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=penguins_clean, 
        x=f'Grupos_{n_clusters}', 
        hue='sex', 
        palette='deep',
        ax=ax
    )
    ax.set_title(f'Distribuição dos Grupos ({n_clusters}) por Sexo')
    ax.set_xlabel('Grupo')
    ax.set_ylabel('Contagem')
    st.pyplot(fig)

# Tabelas cruzadas
st.subheader("Tabelas Cruzadas")

tab1, tab2 = st.tabs(["Por Espécie", "Por Sexo"])

with tab1:
    st.write("**Tabela de contingência - Espécie vs Grupos**")
    cross_species = pd.crosstab(
        penguins_clean["species"],
        penguins_clean[f"Grupos_{n_clusters}"],
        rownames=["Espécies"],
        colnames=[f"{n_clusters} Grupos"],
        margins=True
    )
    st.dataframe(cross_species)

with tab2:
    st.write("**Tabela de contingência - Sexo vs Grupos**")
    cross_sex = pd.crosstab(
        penguins_clean["sex"],
        penguins_clean[f"Grupos_{n_clusters}"],
        rownames=["Sexo"],
        colnames=[f"{n_clusters} Grupos"],
        margins=True
    )
    st.dataframe(cross_sex)

# Estatísticas descritivas
st.header("4) Estatísticas Descritivas por Grupo")
st.write("**Variáveis quantitativas por grupo:**")

# Calcular estatísticas para cada grupo
stats_list = []
for group in sorted(penguins_clean[f'Grupos_{n_clusters}'].unique()):
    group_data = penguins_clean[penguins_clean[f'Grupos_{n_clusters}'] == group]
    group_stats = group_data[penguins_only_numeric_clean.columns].describe().mean(axis=1)
    group_stats['Grupo'] = group
    group_stats['n'] = len(group_data)
    stats_list.append(group_stats)

stats_df = pd.DataFrame(stats_list)
st.dataframe(stats_df)

# Informações gerais
st.sidebar.header("Informações")
st.sidebar.info("""
Esta aplicação demonstra agrupamento hierárquico 
aglomerativo na base de dados de pinguins.

**Variáveis utilizadas:**
- Comprimento do bico
- Profundidade do bico  
- Comprimento da nadadeira
- Massa corporal
""")
