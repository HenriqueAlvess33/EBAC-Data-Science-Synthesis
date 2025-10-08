import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
from io import BytesIO

# Configuração da página
st.set_page_config(page_title="Análise de Dados SINASC", layout="wide")

# Título da aplicação
st.title("📊 Análise de Dados de Nascimento - SINASC")
st.markdown("---")

# Função para plotar gráficos
def plota_pivot_table(dataframe, values, index, funcao, ylabel, xlabel, opcao="nada"):
    fig, ax = plt.subplots(figsize=[15, 5])
    
    if opcao == "nada":
        pd.pivot_table(dataframe, values=values, index=index, aggfunc=funcao).plot(ax=ax)
    elif opcao == "unstack":
        pd.pivot_table(dataframe, values=values, index=index, aggfunc=funcao).unstack().plot(ax=ax)
    elif opcao == "sort_values":
        pd.pivot_table(dataframe, values=values, index=index, aggfunc=funcao).sort_values(values).plot(ax=ax)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    
    return fig

# Dicionário de parâmetros para os gráficos
dicionario_de_parametros = [
    {
        "values": "IDADEMAE",
        "index": "DTNASC",
        "funcao": "count",
        "ylabel": "Quantidade de nascimentos",
        "xlabel": "Datas de nascimentos",
        "opcao": "nada",
        "nome_do_arquivo": "media_idade_mae.png",
        "titulo": "Quantidade de Nascimentos por Data"
    },
    {
        "values": "IDADEMAE",
        "index": "DTNASC",
        "funcao": "mean",
        "ylabel": "Idade das mães",
        "xlabel": "Datas de nascimentos",
        "opcao": "nada",
        "nome_do_arquivo": "contagem_nascimentos_por_data.png",
        "titulo": "Idade Média das Mães por Data de Nascimento"
    },
    {
        "values": "IDADEMAE",
        "index": ["DTNASC", "SEXO"],
        "funcao": "count",
        "ylabel": "Quantidade de nascimentos",
        "xlabel": "Datas de nascimentos",
        "opcao": "unstack",
        "nome_do_arquivo": "contagem_nascimentos_por_data_e_sexo.png",
        "titulo": "Nascimentos por Data e Sexo do Bebê"
    },
    {
        "values": "PESO",
        "index": ["DTNASC", "SEXO"],
        "funcao": "count",
        "ylabel": "Quantidade de nascimentos",
        "xlabel": "Datas de nascimentos",
        "opcao": "unstack",
        "nome_do_arquivo": "peso_recem_nascidos_por_data_e_sexo.png",
        "titulo": "Nascimentos por Data e Sexo (Peso)"
    },
    {
        "values": "PESO",
        "index": "ESCMAE",
        "funcao": "median",
        "ylabel": "Peso do bebê",
        "xlabel": "Tempo de escolaridade das mães",
        "opcao": "nada",
        "nome_do_arquivo": "peso_por_escolaridade.png",
        "titulo": "Peso dos Bebês por Escolaridade da Mãe"
    },
    {
        "values": "APGAR1",
        "index": "GESTACAO",
        "funcao": "mean",
        "ylabel": "Valor de Apgar 1",
        "xlabel": "Semanas de gestação",
        "opcao": "sort_values",
        "nome_do_arquivo": "apgar1_por_gestacao.png",
        "titulo": "Apgar 1 por Tempo de Gestação"
    },
    {
        "values": "APGAR5",
        "index": "GESTACAO",
        "funcao": "mean",
        "ylabel": "Valor de Apgar 5",
        "xlabel": "Semanas de gestação",
        "opcao": "sort_values",
        "nome_do_arquivo": "apgar5_por_gestacao.png",
        "titulo": "Apgar 5 por Tempo de Gestação"
    },
]

# Sidebar para upload de arquivos
st.sidebar.header("📁 Upload de Arquivos")
uploaded_files = st.sidebar.file_uploader(
    "Selecione os arquivos CSV", 
    type=['csv'], 
    accept_multiple_files=True,
    help="Selecione um ou mais arquivos CSV com dados SINASC"
)

# Informações na sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
**Instruções:**
1. Faça upload dos arquivos CSV
2. Os gráficos serão gerados automaticamente
3. Use os controles para navegar entre os gráficos
""")

# Processamento dos arquivos
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} arquivo(s) carregado(s) com sucesso!")
    
    # Seletor de arquivo
    arquivo_selecionado = st.selectbox(
        "Selecione o arquivo para análise:",
        options=[file.name for file in uploaded_files],
        index=0
    )
    
    # Encontrar o arquivo selecionado
    arquivo_atual = next(file for file in uploaded_files if file.name == arquivo_selecionado)
    
    try:
        # Ler o arquivo CSV
        sinasc = pd.read_csv(arquivo_atual)
        
        # Mostrar informações básicas do dataset
        st.subheader("📋 Informações do Dataset")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Registros", len(sinasc))
        
        with col2:
            st.metric("Colunas", len(sinasc.columns))
        
        with col3:
            if 'DTNASC' in sinasc.columns:
                max_data = sinasc.DTNASC.max()[:7] if pd.notna(sinasc.DTNASC.max()) else "N/A"
                st.metric("Data Mais Recente", max_data)
        
        # Mostrar prévia dos dados
        with st.expander("🔍 Visualizar Dados (Primeiras 10 linhas)"):
            st.dataframe(sinasc.head(10))
        
        # Gerar gráficos
        st.subheader("📈 Gráficos de Análise")
        
        # Criar abas para organizar os gráficos
        tabs = st.tabs([param["titulo"] for param in dicionario_de_parametros])
        
        for i, (tab, parametros) in enumerate(zip(tabs, dicionario_de_parametros)):
            with tab:
                # Verificar se as colunas necessárias existem no dataset
                colunas_necessarias = []
                if isinstance(parametros["index"], list):
                    colunas_necessarias.extend(parametros["index"])
                else:
                    colunas_necessarias.append(parametros["index"])
                colunas_necessarias.append(parametros["values"])
                
                colunas_faltantes = [col for col in colunas_necessarias if col not in sinasc.columns]
                
                if colunas_faltantes:
                    st.warning(f"⚠️ Colunas necessárias não encontradas: {', '.join(colunas_faltantes)}")
                    st.info("As colunas disponíveis são:")
                    st.write(list(sinasc.columns))
                else:
                    try:
                        fig = plota_pivot_table(
                            sinasc,
                            values=parametros["values"],
                            index=parametros["index"],
                            funcao=parametros["funcao"],
                            ylabel=parametros["ylabel"],
                            xlabel=parametros["xlabel"],
                            opcao=parametros["opcao"],
                        )
                        st.pyplot(fig)
                        
                        # Botão para download do gráfico
                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                        st.download_button(
                            label=f"📥 Baixar {parametros['titulo']}",
                            data=buf.getvalue(),
                            file_name=parametros["nome_do_arquivo"],
                            mime="image/png",
                            key=f"download_{i}"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao gerar gráfico: {str(e)}")
        
        # Mostrar estatísticas descritivas
        with st.expander("📊 Estatísticas Descritivas"):
            st.dataframe(sinasc.describe())
            
    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {str(e)}")

else:
    # Mensagem inicial quando não há arquivos
    st.info("👆 Faça upload dos arquivos CSV na barra lateral para começar a análise.")
    
    # Exemplo de estrutura esperada
    with st.expander("ℹ️ Estrutura Esperada dos Dados"):
        st.markdown("""
        Os arquivos CSV devem conter as seguintes colunas (ou similares):
        - **DTNASC**: Data de nascimento
        - **IDADEMAE**: Idade da mãe
        - **SEXO**: Sexo do bebê
        - **PESO**: Peso do recém-nascido
        - **ESCMAE**: Escolaridade da mãe
        - **GESTACAO**: Tempo de gestação
        - **APGAR1**: Índice Apgar no 1º minuto
        - **APGAR5**: Índice Apgar no 5º minuto
        """)

# Rodapé
st.markdown("---")
st.caption("Desenvolvido para análise de dados SINASC - Sistema de Informações sobre Nascidos Vivos")