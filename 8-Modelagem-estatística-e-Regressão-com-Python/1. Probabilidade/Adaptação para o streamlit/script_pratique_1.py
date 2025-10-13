import streamlit as st
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as mtick
import pandas as pd
from scipy.stats import norm

# Configuração da página
st.set_page_config(page_title="Probabilidade - Tarefa", layout="wide")
st.title("📊 Probabilidade - Tarefa")

# Sidebar para navegação
st.sidebar.title("Navegação")
section = st.sidebar.radio("Selecione a seção:", [
    "1. Probabilidade Pacote > 15.2kg",
    "2. Percentil 95%",
    "3. CDF - Normal vs t-Student",
    "4. Gráfico CDF",
    "5. Percentis - Normal vs t-Student", 
    "6. Gráfico Percentis"
])

# =============================================================================
# 1. Probabilidade Pacote > 15.2kg
# =============================================================================
if section == "1. Probabilidade Pacote > 15.2kg":
    st.header("1. Probabilidade de pacote > 15.2kg")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Distribuição Normal - Produção de Ração")
        
        # Gera dados
        z = np.random.randn(10000)
        producao_de_racao = z * 0.1 + 15
        
        # Parâmetros
        mu = producao_de_racao.mean()
        variance = 0.1
        sigma = math.sqrt(variance)
        low = 15.2
        
        # Cria figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Gera curva normal
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y = stats.norm.pdf(x, mu, sigma)
        
        # Plota curva
        ax.plot(x, y, linewidth=2, label=f'N(μ={mu:.2f}, σ={sigma:.2f})')
        
        # Preenche área acima de 15.2kg
        plt.fill_between(x, y, where=(low < x), alpha=.5, color='red', 
                        label=f'P(X > {low})')
        
        # Linha vertical em 15.2kg
        ax.axvline(x=low, color='red', linestyle='--', linewidth=2)
        
        # Formatação
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_xlabel('Peso (kg)')
        ax.set_ylabel('Densidade de Probabilidade')
        ax.set_title('Distribuição do Peso dos Pacotes de Ração')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Cálculos")
        
        # Cálculo da probabilidade
        prob_area = 1 - norm.cdf(low, mu, sigma)
        
        st.metric(
            label=f"Probabilidade P(X > {low}kg)",
            value=f"{prob_area:.4f}",
            help="Probabilidade de um pacote pesar mais que 15.2kg"
        )
        
        st.metric(
            label="Média da produção",
            value=f"{procao_de_racao.mean():.4f} kg"
        )
        
        st.metric(
            label="Desvio padrão",
            value=f"{sigma:.4f} kg"
        )
        
        # Estatísticas descritivas
        st.subheader("Estatísticas")
        st.write(f"Pacotes acima de {low}kg na amostra: {len(producao_de_racao[producao_de_racao > 15.2])}")
        st.write(f"Total de pacotes na amostra: {len(producao_de_racao)}")

# =============================================================================
# 2. Percentil 95%
# =============================================================================
elif section == "2. Percentil 95%":
    st.header("2. Percentil de Ordem 95%")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Função de Distribuição Acumulada (CDF)")
        
        # Parâmetros
        mu = 15.0
        variance = 0.1
        sigma = math.sqrt(variance)
        
        # Calcula percentil 95%
        percentil_95 = norm.ppf(0.95, mu, sigma)
        
        # Cria gráfico CDF
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y = norm.cdf(x, mu, sigma)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plota CDF
        ax.plot(x, y, color='blue', linewidth=2, label='CDF')
        
        # Marca percentil 95%
        ax.axvline(x=percentil_95, color='red', linestyle='--', linewidth=2,
                  label=f'Percentil 95% = {percentil_95:.3f}kg')
        ax.axhline(y=0.95, color='green', linestyle='--', linewidth=2,
                  label='95%')
        
        # Formatação
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_xlabel('Peso (kg)')
        ax.set_ylabel('Probabilidade Acumulada')
        ax.set_title('Função de Distribuição Acumulada - Peso dos Pacotes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Resultados")
        
        st.metric(
            label="Percentil 95%",
            value=f"{percentil_95:.4f} kg",
            help="Valor tal que 95% dos pacotes têm peso menor que este"
        )
        
        st.info(
            f"Isso significa que 95% dos pacotes pesam menos que **{percentil_95:.3f} kg** "
            f"e apenas 5% pesam mais que este valor."
        )

# =============================================================================
# 3. CDF - Normal vs t-Student
# =============================================================================
elif section == "3. CDF - Normal vs t-Student":
    st.header("3. CDF - Distribuição Normal vs t-Student")
    
    st.subheader("Probabilidade acumulada P(Y ≤ y)")
    
    # Gera dados
    Y = np.linspace(-6, 6, 200)
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    
    Y_1 = stats.norm.cdf(Y, mu, sigma)  # Normal padrão
    Y_2 = stats.t.cdf(Y, 5)             # t-Student com 5 gl
    
    # Cria DataFrame
    registro = pd.DataFrame({
        "Y": Y,
        "P(Y1 ≤ y) - Normal": Y_1,
        "P(Y2 ≤ y) - t(5)": Y_2,
        "Diferença": Y_2 - Y_1
    })
    
    # Mostra tabela
    st.dataframe(registro.head(10), use_container_width=True)
    
    # Estatísticas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Máxima diferença", f"{registro['Diferença'].max():.4f}")
    with col2:
        st.metric("Mínima diferença", f"{registro['Diferença'].min():.4f}")
    with col3:
        st.metric("Média das diferenças", f"{registro['Diferença'].mean():.4f}")

# =============================================================================
# 4. Gráfico CDF
# =============================================================================
elif section == "4. Gráfico CDF":
    st.header("4. Comparação Gráfica das CDFs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CDFs - Normal vs t-Student")
        
        # Dados
        Y = np.linspace(-6, 6, 200)
        Y_1 = stats.norm.cdf(Y, 0, 1)
        Y_2 = stats.t.cdf(Y, 5)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(Y, Y_1, label='Normal Padrão', linewidth=2)
        ax.plot(Y, Y_2, label='t-Student (5 gl)', linewidth=2)
        
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_xlabel('y')
        ax.set_ylabel('Probabilidade Acumulada P(Y ≤ y)')
        ax.set_title('Funções de Distribuição Acumulada')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Diferença entre CDFs")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plt.plot(Y, Y_2 - Y_1, color="purple", label="CDF t(5) - CDF Normal", linewidth=2)
        plt.axhline(0, color="gray", linestyle="--")
        plt.title("Diferença entre CDFs")
        plt.xlabel("y")
        plt.ylabel("Diferença acumulada")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# =============================================================================
# 5. Percentis - Normal vs t-Student
# =============================================================================
elif section == "5. Percentis - Normal vs t-Student":
    st.header("5. Percentis - Distribuição Normal vs t-Student")
    
    st.subheader("Função Percentil (Inversa da CDF)")
    
    # Gera dados
    Y = np.linspace(0, 1, 20)
    Y_1 = stats.norm.ppf(Y, 0, 1)  # Normal padrão
    Y_2 = stats.t.ppf(Y, 5)        # t-Student com 5 gl
    
    # Cria DataFrame
    registro = pd.DataFrame({
        "Probabilidade": Y,
        "Percentil - Normal": Y_1,
        "Percentil - t(5)": Y_2,
        "Diferença": Y_2 - Y_1
    })
    
    # Mostra tabela
    st.dataframe(registro, use_container_width=True)
    
    # Explicação
    st.info(
        "A função percentil (PPF - Percent Point Function) é a inversa da CDF. "
        "Para uma dada probabilidade p, retorna o valor y tal que P(Y ≤ y) = p."
    )

# =============================================================================
# 6. Gráfico Percentis
# =============================================================================
elif section == "6. Gráfico Percentis":
    st.header("6. Comparação Gráfica dos Percentis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Funções Percentil")
        
        # Dados
        prob = np.linspace(0, 1, 100)
        norm_ppf = stats.norm.ppf(prob, 0, 1)
        t_ppf = stats.t.ppf(prob, 5)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(prob, norm_ppf, label='Normal Padrão', linewidth=2)
        ax.plot(prob, t_ppf, label='t-Student (5 gl)', linewidth=2)
        
        ax.set_xlabel('Probabilidade p')
        ax.set_ylabel('Percentil y (P(Y ≤ y) = p)')
        ax.set_title('Funções Percentil (Inversa da CDF)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Diferença entre Percentis")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plt.plot(prob, t_ppf - norm_ppf, color="purple", 
                label="Percentil t(5) - Percentil Normal", linewidth=2)
        plt.axhline(0, color="gray", linestyle="--")
        plt.title("Diferença entre Funções Percentil")
        plt.xlabel("Probabilidade p")
        plt.ylabel("Diferença nos percentis")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# Informações gerais na sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Sobre")
st.sidebar.info(
    "Esta aplicação demonstra conceitos de probabilidade e estatística "
    "usando distribuições Normal e t-Student."
)