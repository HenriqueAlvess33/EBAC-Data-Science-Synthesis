# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from model_final import CreditScoringModel, create_sample_data, create_sample_model

# Configuração da página
st.set_page_config(
    page_title="Sistema de Credit Scoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-low {
        color: #00d26a;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🏦 Sistema de Credit Scoring</h1>', unsafe_allow_html=True)

# Inicializar o modelo
@st.cache_resource
def load_scoring_model():
    model = CreditScoringModel()
    
    # Verificar se o arquivo do modelo existe
    if not os.path.exists('model_final.pkl'):
        st.warning("Modelo não encontrado. Criando modelo de exemplo...")
        create_sample_model()
    
    success = model.load_model('model_final.pkl')
    if success:
        st.sidebar.success("✅ Modelo carregado com sucesso!")
    else:
        st.sidebar.error("❌ Erro ao carregar modelo")
    
    return model if success else None

# Sidebar
st.sidebar.header("📁 Configurações")

# Upload de arquivo
uploaded_file = st.sidebar.file_uploader(
    "Carregue o arquivo CSV para escoragem",
    type=['csv'],
    help="Selecione o arquivo CSV contendo os dados dos clientes"
)

# Opção para usar dados de exemplo
use_sample_data = st.sidebar.checkbox("Usar dados de exemplo", value=True)

# Informações do modelo
st.sidebar.header("ℹ️ Informações do Modelo")
model = load_scoring_model()

if model:
    model_info = model.get_model_info()
    st.sidebar.write(f"**Tipo:** {model_info['model_type']}")
    st.sidebar.write(f"**Número de Features:** {model_info['n_features']}")
    
    # Mostrar features esperadas
    with st.sidebar.expander("Ver features esperadas"):
        for feature in model_info['features']:
            st.write(f"• {feature}")
else:
    st.sidebar.error("Modelo não carregado corretamente.")

# Requerir CNI para escoragem
st.sidebar.header("🔒 Configurações de Segurança")
require_auth = st.sidebar.checkbox("Requerir autenticação para escoragem", value=False)

if require_auth:
    auth_code = st.sidebar.text_input("Código de autenticação:", type="password")
    if auth_code != "admin123":  # Código simples para demonstração
        st.sidebar.warning("🔒 Autenticação necessária")
        st.stop()

# Conteúdo principal
if uploaded_file is not None or use_sample_data:
    
    # Carregar dados
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado com sucesso! {len(df)} registros encontrados.")
        except Exception as e:
            st.error(f"❌ Erro ao carregar arquivo: {e}")
            st.stop()
    else:
        df = create_sample_data()
        st.info("📋 Usando dados de exemplo para demonstração.")
    
    # Abas principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Visão dos Dados", 
        "🎯 Escoragem", 
        "📊 Análise", 
        "📥 Exportar"
    ])
    
    with tab1:
        st.header("Visualização dos Dados")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Primeiras Linhas")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Informações Gerais")
            st.metric("Total de Registros", len(df))
            st.metric("Total de Colunas", len(df.columns))
            
            # Verificar colunas esperadas
            expected_cols = [
                'sexo', 'posse_de_veiculo', 'posse_de_imovel', 'idade', 
                'tempo_emprego', 'renda', 'qtd_filhos', 'qt_pessoas_residencia'
            ]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Colunas faltantes: {', '.join(missing_cols)}")
            else:
                st.success("✅ Todas as colunas principais presentes")
        
        st.subheader("Estatísticas Descritivas")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.header("Escoragem dos Clientes")
        
        if model is None:
            st.error("❌ Modelo não disponível para escoragem.")
            st.info("💡 Dica: Verifique se o arquivo 'model_final.pkl' existe no diretório.")
        else:
            # Botão para executar escoragem
            if st.button("🎯 Executar Escoragem", type="primary", key="scoring_button"):
                with st.spinner("Processando dados e calculando scores..."):
                    try:
                        # Fazer previsões
                        results_df = model.predict(df)
                        
                        # Mostrar resultados
                        st.success(f"✅ Escoragem concluída! {len(results_df)} clientes processados.")
                        
                        # Métricas resumidas
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_score = results_df['score'].mean()
                            st.metric("Score Médio", f"{avg_score:.3f}")
                        
                        with col2:
                            high_risk_count = len(results_df[results_df['classificacao'] == 'Alto Risco'])
                            st.metric("Alto Risco", high_risk_count)
                        
                        with col3:
                            low_risk_count = len(results_df[results_df['classificacao'] == 'Baixo Risco'])
                            st.metric("Baixo Risco", low_risk_count)
                        
                        with col4:
                            high_risk_pct = (high_risk_count / len(results_df)) * 100
                            st.metric("% Alto Risco", f"{high_risk_pct:.1f}%")
                        
                        # Tabela de resultados
                        st.subheader("Resultados da Escoragem")
                        
                        # Selecionar colunas para mostrar
                        display_cols = []
                        if 'data_ref' in results_df.columns:
                            display_cols.append('data_ref')
                        if 'idade' in results_df.columns:
                            display_cols.append('idade')
                        if 'renda' in results_df.columns:
                            display_cols.append('renda')
                        
                        display_cols.extend(['score', 'classificacao'])
                        
                        # Mostrar todas as colunas originais + resultados
                        results_display = results_df[display_cols].copy()
                        results_display['score'] = results_display['score'].round(4)
                        
                        st.dataframe(results_display, use_container_width=True)
                        
                        # Armazenar resultados na sessão
                        st.session_state['results_df'] = results_df
                        st.session_state['results_display'] = results_display
                        
                    except Exception as e:
                        st.error(f"❌ Erro durante a escoragem: {str(e)}")
                        st.info("💡 Verifique se todas as colunas necessárias estão presentes no arquivo.")
    
    with tab3:
        st.header("Análise dos Resultados")
        
        if 'results_df' not in st.session_state:
            st.info("👆 Execute a escoragem primeiro na aba 'Escoragem'")
        else:
            results_df = st.session_state['results_df']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribuição de scores
                st.subheader("Distribuição de Scores")
                fig_score = px.histogram(
                    results_df, 
                    x='score',
                    nbins=20,
                    title='Distribuição dos Scores de Credit',
                    color_discrete_sequence=['#1f77b4']
                )
                fig_score.add_vline(x=0.068, line_dash="dash", line_color="red", 
                                  annotation_text="Cutoff: 0.068")
                st.plotly_chart(fig_score, use_container_width=True)
            
            with col2:
                # Proporção de classificações
                st.subheader("Classificação de Risco")
                risk_counts = results_df['classificacao'].value_counts()
                fig_pie = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title='Proporção de Classificação de Risco',
                    color=risk_counts.index,
                    color_discrete_map={'Alto Risco': '#ff4b4b', 'Baixo Risco': '#00d26a'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Análise por variáveis
            st.subheader("Análise por Variáveis")
            
            numeric_cols = [col for col in results_df.columns if col not in ['score', 'classificacao', 'prediction'] 
                          and pd.api.types.is_numeric_dtype(results_df[col])]
            
            if numeric_cols:
                analysis_var = st.selectbox(
                    "Selecione a variável para análise:",
                    numeric_cols
                )
                
                if analysis_var:
                    fig_box = px.box(
                        results_df,
                        x='classificacao',
                        y=analysis_var,
                        color='classificacao',
                        title=f'Distribuição de {analysis_var} por Classificação de Risco',
                        color_discrete_map={'Alto Risco': '#ff4b4b', 'Baixo Risco': '#00d26a'}
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
    
    with tab4:
        st.header("Exportar Resultados")
        
        if 'results_df' not in st.session_state:
            st.info("👆 Execute a escoragem primeiro na aba 'Escoragem'")
        else:
            results_df = st.session_state['results_df']
            
            st.subheader("Download dos Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Formato CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"resultados_escoragem_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    type="primary"
                )
            
            with col2:
                # Formato Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    results_df.to_excel(writer, sheet_name='Resultados', index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"resultados_escoragem_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.ms-excel"
                )
            
            # Resumo executivo
            st.subheader("Resumo Executivo")
            
            high_risk_df = results_df[results_df['classificacao'] == 'Alto Risco']
            low_risk_df = results_df[results_df['classificacao'] == 'Baixo Risco']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Clientes de Alto Risco:**")
                if len(high_risk_df) > 0:
                    st.write(f"- Score médio: {high_risk_df['score'].mean():.4f}")
                    if 'idade' in high_risk_df.columns:
                        st.write(f"- Idade média: {high_risk_df['idade'].mean():.1f} anos")
                    if 'renda' in high_risk_df.columns:
                        st.write(f"- Renda média: R$ {high_risk_df['renda'].mean():.2f}")
                else:
                    st.write("Nenhum cliente classificado como alto risco")
            
            with col2:
                st.write("**Clientes de Baixo Risco:**")
                if len(low_risk_df) > 0:
                    st.write(f"- Score médio: {low_risk_df['score'].mean():.4f}")
                    if 'idade' in low_risk_df.columns:
                        st.write(f"- Idade média: {low_risk_df['idade'].mean():.1f} anos")
                    if 'renda' in low_risk_df.columns:
                        st.write(f"- Renda média: R$ {low_risk_df['renda'].mean():.2f}")
                else:
                    st.write("Nenhum cliente classificado como baixo risco")

else:
    # Tela inicial quando nenhum arquivo foi carregado
    st.markdown("""
    ## 👋 Bem-vindo ao Sistema de Credit Scoring
    
    Esta ferramenta permite a escoragem de clientes para análise de risco de crédito utilizando 
    machine learning.
    
    ### 🚀 Como usar:
    
    1. **Carregue seus dados**: Use o menu lateral para fazer upload de um arquivo CSV
    2. **Execute a escoragem**: Na aba "Escoragem", clique no botão para processar os dados
    3. **Analise os resultados**: Visualize distribuições e estatísticas nas abas de análise
    4. **Exporte os resultados**: Faça download dos dados escorados em CSV ou Excel
    
    ### 📋 Estrutura esperada do CSV:
    
    Seu arquivo CSV deve conter as seguintes colunas (ou similares):
    
    - `data_ref`: Data de referência
    - `sexo`: Gênero (M/F)
    - `posse_de_veiculo`: Posse de veículo (Y/N)
    - `posse_de_imovel`: Posse de imóvel (Y/N)
    - `qtd_filhos`: Quantidade de filhos
    - `tipo_renda`: Tipo de renda
    - `educacao`: Nível educacional
    - `estado_civil`: Estado civil
    - `tipo_residencia`: Tipo de residência
    - `idade`: Idade em anos
    - `tempo_emprego`: Tempo no emprego atual
    - `qt_pessoas_residencia`: Pessoas na residência
    - `renda`: Renda mensal
    
    ### 🎯 Ou use dados de exemplo:
    
    Marque a opção "Usar dados de exemplo" no menu lateral para testar a ferramenta.
    """)

# Rodapé
st.markdown("---")
st.markdown(
    "**Desenvolvido para análise de credit scoring** | "
    "Projeto Final - Bootcamp Machine Learning | "
    "v4.0"
)