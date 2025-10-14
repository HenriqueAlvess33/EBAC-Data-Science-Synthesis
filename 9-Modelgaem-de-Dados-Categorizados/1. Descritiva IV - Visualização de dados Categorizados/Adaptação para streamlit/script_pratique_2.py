import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import pickle

# Configuração da página
st.set_page_config(page_title="Análise NFP", layout="wide")

# Título da aplicação
st.title("Análise de Notas Fiscais com Potencial de Crédito")
st.markdown("""
**OBJETIVO:** Prever que tipo de nota tem maior ou menor propensão a fornecer créditos.
""")

# Função IV
def IV(variavel, resposta):
    tab = pd.crosstab(variavel, resposta, margins=True, margins_name="total")
    rótulo_evento = tab.columns[0]
    rótulo_nao_evento = tab.columns[1]
    tab["pct_evento"] = tab[rótulo_evento] / tab.loc["total", rótulo_evento]
    tab["pct_nao_evento"] = tab[rótulo_nao_evento] / tab.loc["total", rótulo_nao_evento]
    tab["woe"] = np.log(
        tab.pct_evento.replace(0, 1e-6) / tab.pct_nao_evento.replace(0, 1e-6)
    )
    tab["iv_parcial"] = (tab.pct_evento - tab.pct_nao_evento) * tab.woe
    return tab["iv_parcial"].sum()

# Função para plotar IV
def plot_iv_comparativo(metadados):
    df_plot = metadados.dropna(subset=["IV"]).sort_values("IV", ascending=False)
    cores = (
        df_plot["IV"]
        .apply(
            lambda x: (
                "grey"
                if x < 0.02
                else (
                    "orange"
                    if x < 0.1
                    else "green" if x < 0.3 else "blue" if x < 0.5 else "purple"
                )
            )
        )
        .tolist()
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(y=df_plot.index, x="IV", data=df_plot, palette=cores, ax=ax)
    ax.axvline(0.02, color="grey", linestyle="--", lw=1, label='0.02 - Inútil')
    ax.axvline(0.1, color="orange", linestyle="--", lw=1, label='0.1 - Fraco')
    ax.axvline(0.3, color="green", linestyle="--", lw=1, label='0.3 - Médio')
    ax.axvline(0.5, color="blue", linestyle="--", lw=1, label='0.5 - Forte')
    ax.set_title("Information Value por variável", fontsize=16)
    ax.set_xlabel("IV")
    ax.set_ylabel("Variável")
    ax.legend()
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig

# Inicializar variáveis de sessão
if 'df_editavel' not in st.session_state:
    st.session_state.df_editavel = None
if 'categorias_disponiveis' not in st.session_state:
    st.session_state.categorias_disponiveis = []

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações da Análise")

# Opções de análise
st.sidebar.subheader("📊 Filtros")
mostrar_raw_data = st.sidebar.checkbox("Mostrar dados brutos", value=False)

# Configurações dos Gráficos
st.sidebar.subheader("🎨 Configurações dos Gráficos")
tema_seaborn = st.sidebar.selectbox(
    "Tema do Seaborn:",
    ["whitegrid", "darkgrid", "white", "dark", "ticks"]
)

# Área para upload de arquivos
st.header("📤 Upload de Dados")
uploaded_file = st.file_uploader(
    "Faça upload do arquivo base_nfp.pkl", 
    type=['pkl'],
    help="Upload do arquivo pickle contendo os dados das notas fiscais"
)

def processar_dados(df):
    """Processa os dados e cria variáveis derivadas"""
    df_editavel = df.copy()
    
    # Dicionário para mapear meses
    dicionario_de_meses = {
        "01": "Janeiro", "02": "Fevereiro", "03": "Março", "04": "Abril",
        "05": "Maio", "06": "Junho", "07": "Julho", "08": "Agosto",
        "09": "Setembro", "10": "Outubro", "11": "Novembro", "12": "Dezembro",
    }

    try:
        # Verificar e converter colunas de data
        for coluna in ['Data Registro', 'Data Emissão']:
            if coluna in df_editavel.columns:
                if not pd.api.types.is_datetime64_any_dtype(df_editavel[coluna]):
                    df_editavel[coluna] = pd.to_datetime(df_editavel[coluna])
        
        # Processar datas
        if 'Data Registro' in df_editavel.columns:
            df_editavel["Mês"] = df_editavel["Data Registro"].dt.month.astype(str).str.zfill(2)
            df_editavel["Mês"] = df_editavel["Mês"].map(dicionario_de_meses)
            df_editavel["Dia da semana"] = df_editavel["Data Registro"].dt.day_name(locale="pt_BR")
            df_editavel["Dia do mês"] = df_editavel["Data Registro"].dt.day
            df_editavel["Fim de semana"] = (
                df_editavel["Data Registro"].dt.weekday.isin([5, 6]).astype(int)
            )
        
        # Criar trimestre
        if 'Data Emissão' in df_editavel.columns:
            df_editavel['Ano_Trimestre'] = df_editavel['Data Emissão'].dt.to_period('Q').astype(str)
            df_editavel['Ano'] = df_editavel['Data Emissão'].dt.year
        elif 'Data Registro' in df_editavel.columns:
            df_editavel['Ano_Trimestre'] = df_editavel['Data Registro'].dt.to_period('Q').astype(str)
            df_editavel['Ano'] = df_editavel['Data Registro'].dt.year
        
        return df_editavel
        
    except Exception as e:
        st.error(f"Erro no processamento dos dados: {str(e)}")
        return df_editavel

if uploaded_file is not None:
    try:
        # Tentar diferentes métodos de carregamento
        with st.spinner("Carregando dados..."):
            file_bytes = uploaded_file.read()
            
            # Método 1: Tentar com pandas
            try:
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_pickle(uploaded_file)
            except:
                # Método 2: Tentar com pickle diretamente
                try:
                    df = pickle.loads(file_bytes)
                except:
                    # Método 3: Tentar com BytesIO
                    try:
                        bytes_io = BytesIO(file_bytes)
                        df = pd.read_pickle(bytes_io)
                    except Exception as e:
                        st.error(f"Erro ao carregar arquivo: {str(e)}")
                        st.stop()
        
        st.success(f"✅ Dados carregados com sucesso! Shape: {df.shape}")
        
        # Processar dados
        with st.spinner("Processando dados..."):
            df_editavel = processar_dados(df)
            st.session_state.df_editavel = df_editavel
            
            # Atualizar categorias disponíveis
            if 'categoria' in df_editavel.columns:
                st.session_state.categorias_disponiveis = df_editavel['categoria'].unique().tolist()
        
        # Mostrar filtros de categoria se disponível
        if st.session_state.categorias_disponiveis:
            st.sidebar.subheader("🎯 Filtros de Categoria")
            categorias_selecionadas = st.sidebar.multiselect(
                "Selecione as categorias:",
                options=st.session_state.categorias_disponiveis,
                default=st.session_state.categorias_disponiveis[:min(3, len(st.session_state.categorias_disponiveis))]
            )
        else:
            categorias_selecionadas = []
            st.sidebar.info("ℹ️ Coluna 'categoria' não encontrada nos dados")

        # Aplicar tema do seaborn
        sns.set_style(tema_seaborn)

        # Abas para organização
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visão Geral", 
            "📈 Análise IV", 
            "📅 Análise Temporal", 
            "🔍 WOE e IV Temporal"
        ])

        with tab1:
            st.header("Visão Geral dos Dados")
            
            # Métricas rápidas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Notas", len(df_editavel))
            with col2:
                if 'flag_credito' in df_editavel.columns:
                    notas_com_credito = df_editavel['flag_credito'].sum()
                    percentual = (notas_com_credito / len(df_editavel)) * 100
                    st.metric("Notas com Crédito", f"{notas_com_credito} ({percentual:.1f}%)")
            with col3:
                if 'Valor NF' in df_editavel.columns:
                    valor_total = df_editavel['Valor NF'].sum()
                    st.metric("Valor Total NF", f"R$ {valor_total:,.0f}")
            with col4:
                if 'Créditos' in df_editavel.columns:
                    creditos_totais = df_editavel['Créditos'].sum()
                    st.metric("Créditos Totais", f"R$ {creditos_totais:,.0f}")
            
            st.subheader("Estrutura dos Dados")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Primeiras Linhas:**")
                st.dataframe(df_editavel.head(10), use_container_width=True)
            
            with col2:
                st.write("**Colunas e Tipos:**")
                info_df = pd.DataFrame({
                    'Coluna': df_editavel.columns,
                    'Tipo': df_editavel.dtypes.values,
                    'Não Nulos': df_editavel.notnull().sum().values
                })
                st.dataframe(info_df, use_container_width=True)
            
            if mostrar_raw_data:
                st.subheader("Dados Completos")
                st.dataframe(df_editavel, use_container_width=True)

        with tab2:
            st.header("Análise do Information Value (IV)")
            
            if 'flag_credito' not in df_editavel.columns:
                st.warning("❌ Coluna 'flag_credito' não encontrada. Não é possível calcular IV.")
            else:
                # Calcular IV para todas as variáveis
                with st.spinner("Calculando Information Values..."):
                    valores_iv = {}
                    metadados = pd.DataFrame(df_editavel.dtypes, columns=['Tipo'])
                    metadados["papel"] = "covariavel"
                    metadados.loc["flag_credito", "papel"] = "resposta"
                    metadados["nunique"] = df_editavel.nunique()

                    variaveis_calculadas = 0
                    for variavel in metadados.index.to_list():
                        if variavel != "flag_credito" and variavel in df_editavel.columns:
                            try:
                                # Verificar se a variável não tem muitos valores únicos
                                if metadados.loc[variavel, 'nunique'] < 100:  # Limite para evitar overfitting
                                    iv = IV(df_editavel[variavel], df_editavel["flag_credito"])
                                    valores_iv[variavel] = iv
                                    variaveis_calculadas += 1
                            except Exception as e:
                                continue

                    if valores_iv:
                        df_iv = pd.DataFrame.from_dict(valores_iv, orient="index", columns=["IV"])
                        df_iv = df_iv.sort_values(by="IV", ascending=False)
                        metadados = metadados.join(df_iv["IV"])
                        metadados["Classificação IV"] = metadados.IV.map(
                            lambda x: (
                                "Inútil" if x <= 0.02
                                else "Fraco" if x <= 0.1
                                else "Médio" if x <= 0.3
                                else "Forte" if x <= 0.5
                                else "Suspeito"
                            )
                        )
                
                if 'IV' in metadados.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("IV por Variável")
                        fig_iv = plot_iv_comparativo(metadados)
                        st.pyplot(fig_iv)
                    
                    with col2:
                        st.subheader("Classificação IV")
                        df_display = (
                            metadados[['IV', 'Classificação IV']]
                            .sort_values('IV', ascending=False)
                            .dropna()
                        )
                        st.dataframe(df_display, use_container_width=True)
                        
                        st.subheader("📊 Resumo das Variáveis")
                        variaveis_fortes = metadados[metadados["Classificação IV"] == "Forte"]
                        if not variaveis_fortes.empty:
                            st.write("**Variáveis com IV Forte:**")
                            for var in variaveis_fortes.index:
                                st.write(f"- {var}: {variaveis_fortes.loc[var, 'IV']:.4f}")
                        else:
                            st.info("Nenhuma variável com IV Forte encontrada")
                else:
                    st.warning("Não foi possível calcular o IV para as variáveis")

        with tab3:
            st.header("Análise Temporal")
            
            if 'Ano' in df_editavel.columns:
                # Criar dataframe para análise temporal
                df_temporal = df_editavel.copy()
                df_temporal['Quantidade de notas'] = 1
                
                # Agrupar por ano
                notas_por_ano = df_temporal.groupby('Ano').agg({
                    'Quantidade de notas': 'count',
                    'Valor NF': 'sum' if 'Valor NF' in df_temporal.columns else None,
                    'Créditos': 'sum' if 'Créditos' in df_temporal.columns else None
                }).reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Evolução da Quantidade de Notas")
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    sns.lineplot(data=notas_por_ano, x="Ano", y="Quantidade de notas", ax=ax1, marker='o')
                    ax1.set_title("Quantidade de Notas por Ano")
                    ax1.set_ylabel("Número de Notas")
                    ax1.grid(True, alpha=0.3)
                    st.pyplot(fig1)
                
                with col2:
                    if 'categoria' in df_temporal.columns:
                        st.subheader("Notas por Categoria")
                        notas_categoria_ano = df_temporal.groupby(['Ano', 'categoria']).size().reset_index(name='Quantidade')
                        
                        fig2, ax2 = plt.subplots(figsize=(12, 6))
                        sns.barplot(data=notas_categoria_ano, x='Ano', y='Quantidade', hue='categoria', ax=ax2)
                        ax2.set_title("Distribuição de Notas por Categoria e Ano")
                        ax2.set_ylabel("Quantidade de Notas")
                        ax2.legend(title="Categoria", bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.tight_layout()
                        st.pyplot(fig2)
                    else:
                        st.info("Coluna 'categoria' não disponível para análise")
            else:
                st.warning("Dados de ano não disponíveis para análise temporal")

        with tab4:
            st.header("WOE e IV Temporal")
            st.info("🔍 Esta análise requer dados temporais e de categoria completos")
            
            if all(col in df_editavel.columns for col in ['categoria', 'Ano', 'flag_credito']):
                try:
                    # Calcular WOE por categoria e ano
                    woedf = df_editavel.groupby(['categoria', 'Ano']).agg({
                        'flag_credito': ['count', 'sum']
                    }).reset_index()
                    
                    woedf.columns = ['categoria', 'Ano', 'total_notas', 'notas_com_credito']
                    woedf['taxa_credito'] = woedf['notas_com_credito'] / woedf['total_notas']
                    
                    # Calcular WOE
                    taxa_media_global = df_editavel['flag_credito'].mean()
                    woedf['WOE'] = np.log(woedf['taxa_credito'] / taxa_media_global)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("WOE por Categoria e Ano")
                        fig3, ax3 = plt.subplots(figsize=(12, 6))
                        sns.lineplot(data=woedf, x='Ano', y='WOE', hue='categoria', ax=ax3, marker='o')
                        ax3.set_title("WOE por Categoria ao Longo do Tempo")
                        ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
                        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.tight_layout()
                        st.pyplot(fig3)
                    
                    with col2:
                        st.subheader("Dados de WOE")
                        st.dataframe(woedf.sort_values(['Ano', 'WOE'], ascending=[True, False]), 
                                   use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Erro ao calcular WOE temporal: {str(e)}")
            else:
                st.warning("Colunas necessárias não encontradas para análise WOE/IV temporal")
                
        # Conclusões
        st.header("📋 Conclusões e Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Principais Insights")
            st.markdown("""
            - **Variáveis-chave**: Identifique as variáveis com maior poder discriminatório
            - **Padrões temporais**: Observe tendências ao longo do tempo
            - **Categorias promissoras**: Descubra quais categorias têm melhor desempenho
            - **Estabilidade**: Analise a consistência dos resultados
            """)
        
        with col2:
            st.subheader("💡 Recomendações")
            st.markdown("""
            - Foque nas variáveis com IV mais alto
            - Monitore categorias com performance consistente
            - Considere fatores temporais nas decisões
            - Valide insights com análise de negócio
            """)
        
    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
        st.info("""
        **🔧 Solução de problemas:**
        - Verifique se o arquivo é um pickle válido do pandas
        - Certifique-se de que o arquivo não está corrompido
        - Tente gerar o arquivo pickle novamente
        - Verifique a versão do pandas utilizada
        """)

else:
    st.info("""
    👆 **Para começar, faça upload do arquivo base_nfp.pkl**
    
    **O arquivo deve conter:** 
    - Dados de notas fiscais
    - Coluna 'flag_credito' para análise
    - Datas de emissão/registro
    - Informações de categoria (opcional)
    """)

# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido para análise de notas fiscais com potencial de crédito**")