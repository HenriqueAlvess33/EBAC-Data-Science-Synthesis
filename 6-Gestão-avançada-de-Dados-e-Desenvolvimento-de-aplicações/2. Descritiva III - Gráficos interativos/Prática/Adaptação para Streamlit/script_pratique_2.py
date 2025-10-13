import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.graph_objs import Layout

# Configuração da página
st.set_page_config(page_title="Análise de Dados Financeiros", layout="wide")

# Título da aplicação
st.title("📊 Dashboard de Análise Financeira")
st.markdown("---")

# Área para upload de arquivos
st.sidebar.header("📁 Upload de Dados")
uploaded_file = st.sidebar.file_uploader(
    "Faça upload do arquivo CSV",
    type=["csv"],
    help="Upload do arquivo com dados financeiros",
)

if uploaded_file is not None:
    try:
        # Carregar os dados
        df = pd.read_csv(uploaded_file, header=[0, 1], index_col=0)
        df = df.round(2)

        # Exibir informações básicas do dataset
        st.sidebar.success("Arquivo carregado com sucesso!")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Período", f"{len(df)} dias")
        with col2:
            st.metric("Ativos", len(df.columns.levels[1]))

        # Mostrar dados brutos
        st.header("📈 Visualização dos Dados")

        expander_dados = st.expander("Visualizar Dados Brutos")
        with expander_dados:
            st.dataframe(df, use_container_width=True)

        # Seleção de ativos para análise
        st.sidebar.header("🔧 Configurações da Análise")

        if "Ativo" in df.columns.names:
            ativos_disponiveis = df.columns.get_level_values(1).unique()
            ativo_selecionado = st.sidebar.selectbox(
                "Selecione o ativo para análise detalhada:", ativos_disponiveis
            )
        else:
            ativo_selecionado = None

        # Cálculo e visualização da Média Móvel
        st.header("📊 Média Móvel de 90 dias")

        try:
            df_mv_90 = df.Close.rolling(window=90).mean()
            df_mv_90 = df_mv_90.stack().reset_index()
            df_mv_90.columns = ["Data", "Ativo", "Média Móvel 90 dias"]

            fig_mv = px.line(
                df_mv_90,
                x="Data",
                y="Média Móvel 90 dias",
                color="Ativo",
                title="Média Móvel de 90 dias",
                labels={"Data": "Data", "Média Móvel 90 dias": "Média Móvel 90 dias"},
                markers=True,
                template="plotly_white",
            )
            fig_mv.update_traces(mode="lines+markers")
            st.plotly_chart(fig_mv, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao calcular média móvel: {e}")

        # Cálculo e visualização do Desvio Padrão
        st.header("📈 Desvio Padrão de 90 dias")

        try:
            df_std_90 = df.Close.rolling(window=90).std()
            df_std_90 = df_std_90.stack().reset_index()
            df_std_90.columns = ["Data", "Ativo", "Desvio Padrão 90 dias"]

            fig_std = px.line(
                df_std_90,
                x="Data",
                y="Desvio Padrão 90 dias",
                color="Ativo",
                title="Desvio Padrão 90 dias",
                labels={
                    "Data": "Data",
                    "Desvio Padrão 90 dias": "Desvio Padrão 90 dias",
                },
                markers=True,
                template="plotly_white",
            )
            fig_std.update_traces(mode="lines+markers")
            st.plotly_chart(fig_std, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao calcular desvio padrão: {e}")

        # Gráfico de Candlestick para ativo específico
        if ativo_selecionado:
            st.header(f"🕯️ Gráfico de Candlestick - {ativo_selecionado}")

            try:
                # Selecionar colunas para o ativo específico
                lista_de_indicadores = [
                    "Close",
                    "High",
                    "Low",
                    "Open",
                    "Volatilidade",
                    "Volume",
                ]
                colunas = [
                    (indicador, ativo_selecionado) for indicador in lista_de_indicadores
                ]

                # Filtrar dados (últimos 60 dias ou dados disponíveis)
                df_ativo = df[colunas].iloc[-60:]
                df_ativo_viz = df_ativo.stack().reset_index()

                # Criar gráfico de candlestick
                graph = go.Candlestick(
                    x=df_ativo_viz.index,
                    open=df_ativo_viz.Open,
                    close=df_ativo_viz.Close,
                    high=df_ativo_viz.High,
                    low=df_ativo_viz.Low,
                    name=ativo_selecionado,
                    showlegend=True,
                )

                layout = Layout(
                    title=f"Gráfico de Candlestick - {ativo_selecionado}",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                grafico = go.Figure(data=[graph], layout=layout)
                grafico.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1)
                grafico.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1)

                st.plotly_chart(grafico, use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao criar gráfico de candlestick: {e}")

        # Informações adicionais
        st.sidebar.header("ℹ️ Informações")
        st.sidebar.info(
            """
        Esta aplicação permite:
        - Visualizar dados financeiros
        - Calcular médias móveis
        - Analisar volatilidade
        - Gerar gráficos de candlestick
        """
        )

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        st.info(
            "Verifique se o formato do arquivo está correto (header duplo, índice temporal)"
        )

else:
    # Mensagem inicial quando não há arquivo carregado
    st.info("👆 Faça upload de um arquivo CSV para começar a análise")

    # Exemplo de estrutura esperada
    st.subheader("Estrutura esperada do arquivo CSV:")
    st.code(
        """
    Colunas com header duplo (MultiIndex):
    - Primeiro nível: Indicadores (Close, High, Low, Open, etc.)
    - Segundo nível: Nome dos ativos (IBM, AAPL, etc.)
    - Índice: Datas (formato datetime)
    """
    )

    # Colunas de exemplo
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Exemplo de estrutura:")
        exemplo_df = pd.DataFrame(
            {
                ("Close", "IBM"): [150.2, 151.5, 149.8],
                ("Close", "AAPL"): [180.3, 182.1, 179.5],
                ("High", "IBM"): [151.0, 152.2, 150.5],
                ("High", "AAPL"): [181.5, 183.0, 180.8],
            },
            index=["2024-01-01", "2024-01-02", "2024-01-03"],
        )
        st.dataframe(exemplo_df)
