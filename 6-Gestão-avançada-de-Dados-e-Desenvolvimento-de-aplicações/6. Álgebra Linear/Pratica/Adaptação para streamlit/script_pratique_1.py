import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Álgebra Linear - Análise de Notas", layout="wide")

# Título da aplicação
st.title("📊 Análise de Notas - Álgebra Linear")
st.markdown("""
Esta aplicação demonstra transformações lineares e correlações entre notas de duas provas (p1 e p2).
""")

# Sidebar para configurações
st.sidebar.header("Configurações")

# Configurações no sidebar
st.sidebar.subheader("Parâmetros dos Dados")
num_alunos = st.sidebar.slider("Número de alunos", min_value=10, max_value=100, value=50)
ruido = st.sidebar.slider("Nível de ruído", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
seed = st.sidebar.number_input("Seed para reproducibilidade", min_value=1, max_value=1000, value=123)

st.sidebar.subheader("Configurações do Gráfico")
cor_original = st.sidebar.color_picker("Cor dos dados originais", "#FF0000")
cor_transformado = st.sidebar.color_picker("Cor dos dados transformados", "#0000FF")
tamanho_ponto = st.sidebar.slider("Tamanho dos pontos", min_value=10, max_value=100, value=50)

# Geração dos dados com parâmetros do sidebar
np.random.seed(seed)

p1 = np.random.random(num_alunos) * 10
p1[p1 > 10] = 10
p1[p1 < 0] = 0

p2 = p1 + np.random.normal(0, ruido, num_alunos)
p2[p2 > 10] = 10
p2[p2 < 0] = 0

# Criação do DataFrame
df = pd.DataFrame({'p1': p1, 'p2': p2})
df['média'] = df.mean(axis=1)
df['evolução'] = df['média'] - df['p1']
df['soma'] = df[['p1', 'p2']].sum(axis=1)

# Função para plotar transformações lineares (adaptada para Streamlit)
def plot_transf_linear(m, df, t='padrão', title="Transformação Linear"):
    """
    Realiza uma transformação linear e plota os dados transformados.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Configuração do gráfico
    xmin, xmax, ymin, ymax = -15, 15, -15, 15
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_aspect("equal", "box")
    ax.set_xticks(range(xmin, xmax + 1, 5))
    ax.set_yticks(range(ymin, ymax + 1, 5))
    ax.axvline(0, linewidth=0.5, color='black', linestyle='--')
    ax.axhline(0, linewidth=0.5, color='black', linestyle='--')
    ax.grid(True, alpha=0.3)

    # Desenha as setas representando as colunas da matriz de transformação
    if t == 'padrão':
        ax.arrow(0, 0, m[0, 0], m[1, 0], head_width=0.5, head_length=0.1, 
                fc="blue", ec="blue", length_includes_head=True, label='Vetor 1')
        ax.arrow(0, 0, m[1, 0], m[1, 1], head_width=0.5, head_length=0.1, 
                fc="red", ec="red", length_includes_head=True, label='Vetor 2')
    elif t == 'transposta':
        ax.arrow(0, 0, m[0, 0], m[0, 1], head_width=0.5, head_length=0.1, 
                fc="blue", ec="blue", length_includes_head=True, label='Vetor 1')
        ax.arrow(0, 0, m[1, 0], m[1, 1], head_width=0.5, head_length=0.1, 
                fc="red", ec="red", length_includes_head=True, label='Vetor 2')

    # Plota os dados originais e transformados
    ax.scatter(df['p1'], df['p2'], color=cor_original, alpha=0.7, label='Originais (p1, p2)', s=tamanho_ponto)
    ax.scatter(df['média'], df['evolução'], color=cor_transformado, alpha=0.7, 
               label='Transformados (média, evolução)', s=tamanho_ponto)

    ax.set_xlabel("p1 / Média")
    ax.set_ylabel("p2 / Evolução")
    ax.set_title(title)
    ax.legend()
    
    return fig

# Matrizes de transformação
matriz_de_transformacao = np.array([
    [0.5, 0.5],  # Calcula a média
    [-1, 1],     # Calcula a diferença
])

matriz_de_soma_e_diferenca = np.array([
    [1, 1],   # Soma p1 e p2
    [1, -1],  # Diferença entre p1 e p2
])

# Layout em colunas
col1, col2 = st.columns(2)

with col1:
    st.header("1. Dados das Notas")
    
    # Mostrar dataframe
    st.subheader("DataFrame com as Notas")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Estatísticas descritivas
    st.subheader("Estatísticas Descritivas")
    st.dataframe(df[['p1', 'p2', 'média']].describe(), use_container_width=True)

with col2:
    st.header("2. Correlações")
    
    # Cálculo das correlações
    corr_p1_p2 = df[['p1', 'p2']].corr().iloc[0, 1]
    corr_media_evolucao = df[['média', 'evolução']].corr().iloc[0, 1]
    corr_soma_evolucao = df[['soma', 'evolução']].corr().iloc[0, 1]
    
    # Exibição das correlações
    st.metric("Correlação entre p1 e p2", f"{corr_p1_p2:.3f}")
    st.metric("Correlação entre média e evolução", f"{corr_media_evolucao:.3f}")
    st.metric("Correlação entre soma e evolução", f"{corr_soma_evolucao:.3f}")
    
    # Análise das correlações
    st.subheader("Análise das Correlações")
    st.markdown("""
    - **p1 e p2**: Alta correlação positiva (esperado, pois p2 é derivada de p1)
    - **média e evolução**: Baixa correlação (transformação remove dependência linear)
    - **soma e evolução**: Correlação moderada (relação matemática direta)
    """)

# Visualizações
st.header("3. Visualizações das Transformações Lineares")

# Gráfico 1: Transformação média/diferença
col3, col4 = st.columns(2)

with col3:
    st.subheader("Transformação: Média e Diferença")
    fig1 = plot_transf_linear(matriz_de_transformacao, df=df, t='transposta', 
                             title="Transformação: Média e Diferença")
    st.pyplot(fig1)
    
    st.markdown("""
    **Matriz de transformação:**
    ```
    [0.5, 0.5]  → Média
    [-1,  1 ]  → Diferença
    ```
    """)

with col4:
    st.subheader("Transformação: Soma e Diferença")
    fig2 = plot_transf_linear(matriz_de_soma_e_diferenca, df=df, t='transposta', 
                             title="Transformação: Soma e Diferença")
    st.pyplot(fig2)
    
    st.markdown("""
    **Matriz de transformação:**
    ```
    [1,  1]  → Soma
    [1, -1]  → Diferença
    ```
    """)

# Gráfico de dispersão adicional
st.header("4. Gráficos de Dispersão Comparativos")

col5, col6 = st.columns(2)

with col5:
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(df['p1'], df['p2'], alpha=0.7, color=cor_original, s=tamanho_ponto)
    ax3.set_xlabel('p1')
    ax3.set_ylabel('p2')
    ax3.set_title('Dados Originais: p1 vs p2')
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

with col6:
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.scatter(df['média'], df['evolução'], alpha=0.7, color=cor_transformado, s=tamanho_ponto)
    ax4.set_xlabel('Média')
    ax4.set_ylabel('Evolução')
    ax4.set_title('Dados Transformados: Média vs Evolução')
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

# Explicação matemática
st.header("5. Explicação Matemática")
st.markdown("""
### Transformações Lineares Aplicadas

**1. Transformação para Média e Diferença:**
média = 0.5*p1 + 0.5*p2
diferença = -p1 + p2


**2. Transformação para Soma e Diferença:**
soma = p1 + p2
diferença = p1 - p2

### Por que reduzimos a correlação?
- A transformação para **média e diferença** cria variáveis menos correlacionadas
- A **média** captura a tendência central
- A **diferença** captura a variação relativa
- Isso é útil em análise de dados e machine learning para features menos redundantes
""")

# Rodapé
st.markdown("---")
st.markdown("*Aplicação adaptada de notebook para Streamlit - Álgebra Linear*")