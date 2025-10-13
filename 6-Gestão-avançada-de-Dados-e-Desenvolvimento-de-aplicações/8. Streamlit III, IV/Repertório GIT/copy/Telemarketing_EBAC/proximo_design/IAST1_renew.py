import streamlit as st
import pandas as pd
import plotly.express as px

# ===== CONFIGURAÇÃO INICIAL =====
st.set_page_config(page_title="ATARI - Análise de Dados", page_icon="📊", layout="wide")

# ===== CSS PERSONALIZADO =====
st.markdown(
    """
<style>
    .stApp { background-color: #f5f5f5; }
    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    .stButton>button { background-color: #3498db; border-radius: 5px; }
    .stDataFrame { box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
</style>
""",
    unsafe_allow_html=True,
)

# ===== SIDEBAR (Filtros) =====
with st.sidebar:
    st.title("⚙️ Filtros")
    uploaded_file = st.file_uploader(
        "Upload CSV", type="csv", help="Arraste um arquivo CSV aqui"
    )
    profissao = st.multiselect(
        "Profissões", ["Services", "Technician", "Admin", "Engineer"]
    )
    st.color_picker("Cor do Tema", "#3498db")

# ===== LAYOUT PRINCIPAL =====
st.title("📈 ATARI - Análise de Dados Bancários")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Abas para organizar conteúdo
    tab1, tab2, tab3 = st.tabs(["📊 Visualização", "📑 Dados Brutos", "📌 Métricas"])

    with tab1:
        # Gráfico interativo
        fig = px.bar(df, x="job", y="balance", color="job", title="Saldo por Profissão")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Tabela com paginação
        st.dataframe(df.style.highlight_max(axis=0, color="#3498db"), height=400)

    with tab3:
        # Métricas em cards
        col1, col2 = st.columns(2)
        col1.metric("Total de Registros", len(df))
        col2.metric("Saldo Médio", f"R$ {df['balance'].mean():.2f}")

else:
    st.warning("⏳ Faça upload de um arquivo CSV para começar!")

# ===== FOOTER =====
st.divider()
st.caption("Desenvolvido com Streamlit 🚀")
