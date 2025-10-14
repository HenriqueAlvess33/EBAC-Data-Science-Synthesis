import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io

# Configuração da página
st.set_page_config(page_title="Análise NFP - AMA", page_icon="📊", layout="wide")

# Título e descrição
st.title("Análise de Nota Fiscal Paulista - AMA")
st.markdown("""
**OBJETIVO:** Analisar a propensão de notas fiscais paulistas gerarem créditos para doação à AMA.

Esta aplicação processa dados da NFP e calcula métricas como WOE e Information Value para prever 
quais tipos de notas têm maior probabilidade de retorno.
""")

# Área de upload de arquivos
st.header("Upload de Dados")
uploaded_file = st.file_uploader(
    "Faça upload do arquivo base_nfp.pkl", 
    type=['pkl'], 
    help="Carregue o arquivo base_nfp.pkl contendo os dados da Nota Fiscal Paulista"
)

if uploaded_file is not None:
    try:
        # Carregar os dados
        df = pd.read_pickle(uploaded_file)
        
        st.success("Arquivo carregado com sucesso!")
        
        # Mostrar informações básicas do dataset
        st.subheader("Pré-visualização dos Dados")
        st.write(f"Shape do dataset: {df.shape}")
        st.dataframe(df.head())
        
        # Filtro de data
        st.subheader("Filtros")
        st.write("Considerando somente dados de janeiro de 2020 em diante:")
        df_2020 = df.loc[df['Data Emissão'] >= '2020-01-01']
        st.write(f"Registros após 2020: {len(df_2020)}")
        
        # Análise da probabilidade de retorno
        st.header("1. Análise da Probabilidade de Retorno")
        
        # Tabela de contingência
        tab = pd.crosstab(df_2020['categoria'], df_2020['flag_credito'], 
                         margins=True, margins_name='Total')
        rotulo_evento = tab[1]
        rotulo_nao_evento = tab[0]
        tab['pct_evento'] = rotulo_evento / tab['Total']
        tab['pct_nao_evento'] = rotulo_nao_evento / tab['Total']
        tab['chances_evento'] = rotulo_nao_evento / rotulo_evento
        
        st.subheader("Tabela de Contingência")
        st.dataframe(tab)
        
        # Gráfico de proporções
        st.subheader("Proporção de Retorno por Categoria")
        
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        tab_sorted = tab[:-1].sort_values('pct_evento', ascending=True)  # Excluir linha 'Total'
        
        bars = ax1.barh(tab_sorted.index, tab_sorted['pct_evento'] * 100, 
                       color='skyblue', edgecolor='black')
        ax1.set_xlabel('Percentual de Retorno (%)')
        ax1.set_ylabel('Categoria')
        ax1.set_title('Proporção de Notas com Retorno > 0 por Categoria')
        
        # Adicionar valores nas barras
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center')
        
        st.pyplot(fig1)
        
        # Cálculo do WOE
        st.header("2. Cálculo do WOE (Weight of Evidence)")
        
        tab['WOE'] = np.log(tab.chances_evento)
        
        st.subheader("Tabela com WOE")
        st.dataframe(tab)
        
        # Gráfico WOE
        st.subheader("Gráfico WOE por Categoria")
        
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        tab_woe_sorted = tab[:-1].sort_values('WOE', ascending=True)  # Excluir linha 'Total'
        
        colors = ['red' if x < 0 else 'green' for x in tab_woe_sorted['WOE']]
        bars = ax2.barh(tab_woe_sorted.index, tab_woe_sorted['WOE'], 
                       color=colors, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('WOE')
        ax2.set_ylabel('Categoria')
        ax2.set_title('WOE por Categoria (Evento: retorno > 0)')
        ax2.grid(axis='x', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + (0.01 if width >= 0 else -0.01), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', 
                    ha='left' if width >= 0 else 'right', 
                    va='center',
                    fontweight='bold')
        
        st.pyplot(fig2)
        
        # Cálculo do Information Value
        st.header("3. Cálculo do Information Value (IV)")
        
        tab['iv_parcial'] = (tab.pct_evento - tab.pct_nao_evento) * tab['WOE']
        iv_total = tab.iv_parcial.sum()
        
        st.metric("Information Value Total", f"{iv_total:.4f}")
        
        # Interpretação do IV
        st.subheader("Interpretação do IV")
        if iv_total < 0.02:
            st.warning("IV < 0.02: A variável praticamente não tem poder preditivo")
        elif iv_total < 0.1:
            st.info("IV entre 0.02 e 0.1: A variável tem poder preditivo fraco")
        elif iv_total < 0.3:
            st.success("IV entre 0.1 e 0.3: A variável tem poder preditivo médio")
        else:
            st.success("IV > 0.3: A variável tem forte poder preditivo")
        
        # Conclusões
        st.header("4. Conclusões")
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #6A5ACD;'>
        <font color='#6A5ACD'><b>Conclusão:</b> A partir da análise da tabela e do gráfico de barras, conclui-se que os cidadãos devem direcionar seu consumo — e contribuir com ONGs de caridade — 
        principalmente por meio dos setores de vestuário, auto posto e varejo, que apresentaram os maiores índices de retorno. Em contraste, restaurantes mostraram baixo retorno e devem ser evitados para esse objetivo. 
        No entanto, segundo a lógica de Naeem Siddiqi, a variável 'categoria', que fundamenta essa análise, poderia ser considerada inútil por apresentar um índice de informação negativo.</font>
        </div>
        """, unsafe_allow_html=True)
        
        # Download dos resultados
        st.header("5. Exportar Resultados")
        
        # Converter tabela para CSV
        csv = tab.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download da Tabela de Análise (CSV)",
            data=csv,
            file_name="analise_nfp_resultados.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {str(e)}")
        st.info("Certifique-se de que o arquivo é um pickle válido do pandas com a estrutura correta.")
else:
    st.info("👆 Por favor, faça upload do arquivo base_nfp.pkl para iniciar a análise.")
    
# Informações adicionais na sidebar
with st.sidebar:
    st.header("ℹ️ Informações")
    st.markdown("""
    **Sobre os dados:**
    - Base da Nota Fiscal Paulista
    - Dados de doação automática para a AMA
    - Período: histórico completo
    
    **Campos principais:**
    - CNPJ emit. / Emitente
    - Valor NF / Créditos
    - Data Emissão / Data Registro
    - Situação do Crédito
    - Categoria
    - flag_credito
    """)
    
    st.header("📊 Métricas Calculadas")
    st.markdown("""
    - **WOE**: Weight of Evidence
    - **IV**: Information Value
    - Proporções de retorno
    - Tabelas de contingência
    """)