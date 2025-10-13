import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Cálculo Integral - x³", page_icon="📊", layout="wide")

# Título principal
st.title("Cálculo - Tarefa 02")
st.markdown("""
1. Calcule de forma numérica a integral da função $x^3$ avaliada entre os pontos $1$ e $2$
2. Calcule essa área de forma analítica
""")

# Define a função f(x) como x^3
f = lambda x: np.power(x, 3)

# Função para calcular a área aproximada usando a soma de Riemann
def calcula_area(a, b, func, n_retangulos):
    bins = n_retangulos
    delta = (b - a) / bins
    x_cols = np.linspace(a + delta/2, b - delta/2, bins)
    ycols = func(x_cols)
    return sum(delta * ycols)

# Função para criar o gráfico da aproximação de Riemann
def criar_grafico_riemann(a, b, func, n_retangulos):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(a, b)
    ax.set_ylim(0, 20)
    
    # Plota a curva da função
    x = np.linspace(a, b, 1000)
    y = func(x)
    ax.plot(x, y, "-", color='blue', label='f(x) = x³', linewidth=2)
    
    # Configurações do gráfico
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Calcula e plota os retângulos
    bins = n_retangulos
    delta = (b - a) / bins
    x_cols = np.linspace(a + delta/2, b - delta/2, bins)
    ycols = func(x_cols)
    
    # Desenha os retângulos
    ax.bar(x_cols, ycols, width=delta, align='center', alpha=0.5, 
           color='red', edgecolor='darkred', label='Retângulos de Riemann')
    
    # Calcula a área
    area = calcula_area(a, b, func, n_retangulos)
    
    ax.set_title(f"Aproximação de Riemann - {n_retangulos} retângulos\nÁrea Aproximada: {area:.6f}")
    ax.legend()
    
    return fig, area

# Sidebar para controles
st.sidebar.header("Controles")
n_retangulos = st.sidebar.slider(
    "Número de retângulos", 
    min_value=1, 
    max_value=100, 
    value=10,
    help="Aumente o número de retângulos para melhorar a precisão"
)

# Layout em colunas
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualização da Aproximação de Riemann")
    
    # Cria e exibe o gráfico
    fig, area_aproximada = criar_grafico_riemann(1, 2, f, n_retangulos)
    st.pyplot(fig)

with col2:
    st.subheader("Resultados")
    
    # Resultado numérico
    st.metric(
        label=f"Área Aproximada ({n_retangulos} retângulos)",
        value=f"{area_aproximada:.6f}"
    )
    
    # Cálculo analítico
    st.subheader("Cálculo Analítico")
    st.latex(r"f(x) = x^3")
    st.latex(r"\int_1^2 x^3\,dx = \left[\frac{x^4}{4}\right]_1^2")
    st.latex(r"= \frac{2^4}{4} - \frac{1^4}{4} = \frac{16}{4} - \frac{1}{4} = 4 - 0.25 = 3.75")
    
    area_exata = 3.75
    st.metric(
        label="Área Exata (Analítica)",
        value=f"{area_exata:.6f}"
    )
    
    # Erro
    erro = abs(area_exata - area_aproximada)
    st.metric(
        label="Erro",
        value=f"{erro:.6f}"
    )

# Tabela comparativa para diferentes números de retângulos
st.subheader("Comparação para Diferentes Números de Retângulos")

# Calcula áreas para diferentes valores de n
valores_n = [1, 5, 10, 20, 50, 100]
areas = []
erros = []

for n in valores_n:
    area = calcula_area(1, 2, f, n)
    erro = abs(3.75 - area)
    areas.append(area)
    erros.append(erro)

# Cria tabela
import pandas as pd
df_comparacao = pd.DataFrame({
    'Nº Retângulos': valores_n,
    'Área Aproximada': [f"{a:.6f}" for a in areas],
    'Erro': [f"{e:.6f}" for e in erros]
})

st.dataframe(df_comparacao, use_container_width=True)

# Explicação matemática
st.subheader("Explicação Matemática")
st.markdown("""
**Método Numérico (Soma de Riemann):**

- Dividimos o intervalo [1, 2] em `n` subintervalos iguais
- Cada subintervalo tem largura: $\Delta x = \\frac{2-1}{n} = \\frac{1}{n}$
- Usamos o ponto médio de cada subintervalo: $x_i^* = 1 + \\left(i - \\frac{1}{2}\\right)\\Delta x$
- A área aproximada é: $\\sum_{i=1}^n f(x_i^*) \\cdot \\Delta x$

**Método Analítico:**

Usamos o Teorema Fundamental do Cálculo para calcular a integral definida exata.
""")

# Rodapé
st.markdown("---")
st.caption("Adaptado de notebook para Streamlit - Visualização Interativa da Integral de x³")