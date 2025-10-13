import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(page_title="Análise de Volatilidade", layout="wide")

# Título da aplicação
st.title("📊 Análise de Volatilidade de Ativos")


# Dados de exemplo para quando o yfinance não funciona
def criar_dados_exemplo(ativos, data_inicial):
    """Cria dados de exemplo para demonstração"""
    dates = pd.date_range(start=data_inicial, end=datetime.now(), freq="D")
    dados = {}

    for ativo in ativos:
        # Preços base realistas para cada ativo
        if ativo == "AAPL":
            base_price = 150
        elif ativo == "MSFT":
            base_price = 300
        elif ativo == "GOOGL":
            base_price = 120
        elif ativo == "TSLA":
            base_price = 200
        else:
            base_price = 100

        # Gerar dados realistas com alguma volatilidade
        high_prices = []
        low_prices = []

        for i in range(len(dates)):
            # Preço base com tendência leve
            base = base_price + (i * 0.1)
            # Alta e baixa com volatilidade realista
            high = base * (1 + 0.02 + 0.01 * (i % 10))
            low = base * (1 - 0.015 - 0.008 * (i % 8))
            high_prices.append(round(high, 2))
            low_prices.append(round(low, 2))

        dados[("High", ativo)] = high_prices
        dados[("Low", ativo)] = low_prices

    df = pd.DataFrame(dados, index=dates)
    return df


# Sidebar para upload e configurações
with st.sidebar:
    st.header("Configurações")

    # Opção: usar dados pré-definidos ou fazer upload
    opcao_dados = st.radio(
        "Escolha a fonte dos dados:",
        ["Usar dados de exemplo", "Fazer upload de arquivo CSV"],
    )

    if opcao_dados == "Usar dados de exemplo":
        ativos_input = st.text_input(
            "Digite os tickers dos ativos (separados por vírgula):",
            value="AAPL, MSFT, GOOGL",
        )
        ativos = [
            ativo.strip().upper() for ativo in ativos_input.split(",") if ativo.strip()
        ]

        # Data mais recente para garantir que há dados
        data_inicial = st.date_input(
            "Data inicial:", value=datetime.now() - timedelta(days=90)  # 3 meses atrás
        )

        # Botão para carregar dados
        carregar_dados = st.button("Carregar Dados de Exemplo")

    else:  # Upload de arquivo
        st.subheader("Upload de Arquivo")
        arquivo = st.file_uploader(
            "Faça upload do arquivo CSV com dados dos ativos:",
            type=["csv"],
            help="O arquivo deve conter colunas para preços High e Low",
        )
        carregar_dados = arquivo is not None

# Área principal da aplicação
if carregar_dados:

    if opcao_dados == "Usar dados de exemplo":
        if not ativos:
            st.error("Por favor, digite pelo menos um ticker de ativo.")
            st.stop()

        # Usar dados de exemplo
        with st.spinner("Gerando dados de exemplo..."):
            try:
                df_ativos = criar_dados_exemplo(ativos, data_inicial)

                if df_ativos.empty:
                    st.error("Erro ao gerar dados de exemplo.")
                    st.stop()

                st.success(
                    f"""
                ✅ Dados de exemplo gerados com sucesso!
                - Tickers: {ativos}
                - Período: {len(df_ativos)} dias
                - Data inicial: {df_ativos.index[0].strftime('%Y-%m-%d')}
                - Data final: {df_ativos.index[-1].strftime('%Y-%m-%d')}
                """
                )

                st.info(
                    "💡 Estes são dados simulados para demonstração. Para dados reais, use a opção de upload com arquivos CSV."
                )

            except Exception as e:
                st.error(f"Erro ao gerar dados: {e}")
                st.stop()

    else:  # Usar dados do upload
        try:
            df_ativos = pd.read_csv(arquivo)
            st.success(f"✅ Arquivo carregado com sucesso! {len(df_ativos)} registros.")

            # Tentar converter coluna de data se existir
            date_columns = ["Date", "Data", "DATA", "date", "Datetime", "datetime"]
            for col in date_columns:
                if col in df_ativos.columns:
                    df_ativos[col] = pd.to_datetime(df_ativos[col])
                    df_ativos.set_index(col, inplace=True)
                    break

        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {e}")
            st.stop()

    # Exibindo dados brutos
    st.subheader("📈 Dados dos Ativos (Primeiras 10 linhas)")
    st.dataframe(df_ativos.head(10).round(2), use_container_width=True)

    # Processamento para visualização
    try:
        # Verificar a estrutura do DataFrame
        st.subheader("🔍 Estrutura dos Dados")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Colunas disponíveis:**")
            st.write(list(df_ativos.columns))

        with col2:
            st.write(f"**Forma do DataFrame:** {df_ativos.shape}")
            st.write(f"**Período:** {len(df_ativos)} dias")

        # Cálculo da volatilidade - abordagem mais robusta
        if isinstance(df_ativos.columns, pd.MultiIndex):
            # DataFrame com multi-index (formato yfinance)
            st.info("📊 Estrutura multi-index detectada")

            # Processar dados para o gráfico - CORREÇÃO AQUI
            temp_df = df_ativos[["High", "Low"]].stack().reset_index()
            temp_df["Volatilidade"] = (temp_df["High"] - temp_df["Low"]).abs().round(2)
            temp_df.rename(
                columns={"level_1": "Ticker", "level_0": "Date"}, inplace=True
            )  # CORREÇÃO

        else:
            # DataFrame com colunas simples
            st.info("📊 Estrutura de colunas simples detectada")

            # Verificar se temos colunas High e Low
            if "High" not in df_ativos.columns or "Low" not in df_ativos.columns:
                st.error("❌ Colunas 'High' e 'Low' não encontradas no arquivo.")
                st.stop()

            # Criar estrutura similar para visualização
            temp_df = df_ativos[["High", "Low"]].copy()
            temp_df["Ticker"] = "Ativo"  # Nome padrão para único ativo
            temp_df["Volatilidade"] = (temp_df["High"] - temp_df["Low"]).abs().round(2)

            # Verificar se há coluna de data no índice
            if df_ativos.index.name is not None:
                temp_df["Date"] = df_ativos.index
            else:
                temp_df["Date"] = temp_df.index  # Usar índice como data

        # Exibindo dados processados
        st.subheader("📊 Dados de Volatilidade Processados")
        st.dataframe(temp_df.head(10), use_container_width=True)

        # Gráfico de volatilidade
        st.subheader("📈 Volatilidade Diária")

        # CORREÇÃO: Verificar se a coluna Date existe
        if "Date" not in temp_df.columns:
            st.error("❌ Coluna 'Date' não encontrada nos dados processados.")
            st.write("Colunas disponíveis:", list(temp_df.columns))
        else:
            fig = px.line(
                temp_df,
                x="Date",
                y="Volatilidade",
                color="Ticker",
                title="Volatilidade Diária - Variação entre Preço Máximo e Mínimo",
                labels={"Date": "Data", "Volatilidade": "Volatilidade (R$)"},
                markers=True,
                template="plotly_white",
            )
            fig.update_traces(mode="lines+markers")
            fig.update_layout(
                height=500,
                xaxis_title="Data",
                yaxis_title="Volatilidade (R$)",
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)

        # Estatísticas resumidas
        st.subheader("📋 Estatísticas de Volatilidade")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            volatilidade_media = temp_df["Volatilidade"].mean()
            st.metric("Volatilidade Média", f"R$ {volatilidade_media:.2f}")

        with col2:
            volatilidade_max = temp_df["Volatilidade"].max()
            st.metric("Volatilidade Máxima", f"R$ {volatilidade_max:.2f}")

        with col3:
            volatilidade_min = temp_df["Volatilidade"].min()
            st.metric("Volatilidade Mínima", f"R$ {volatilidade_min:.2f}")

        with col4:
            dias_analisados = (
                temp_df["Date"].nunique() if "Date" in temp_df.columns else len(temp_df)
            )
            st.metric("Dias Analisados", dias_analisados)

        # Tabela de estatísticas por ativo
        if "Ticker" in temp_df.columns:
            st.subheader("📊 Estatísticas por Ativo")
            stats_por_ativo = (
                temp_df.groupby("Ticker")["Volatilidade"]
                .agg(
                    [
                        ("Média", "mean"),
                        ("Máxima", "max"),
                        ("Mínima", "min"),
                        ("Desvio Padrão", "std"),
                        ("Dias", "count"),
                    ]
                )
                .round(2)
            )

            st.dataframe(stats_por_ativo, use_container_width=True)

        # Download dos dados processados
        st.subheader("💾 Download dos Dados")

        # Criando diretório de output
        os.makedirs("./output", exist_ok=True)

        # Salvando o arquivo
        nome_arquivo = "./output/dados_volatilidade_processados.csv"
        temp_df.to_csv(nome_arquivo, index=False)

        col1, col2 = st.columns(2)

        with col1:
            with open(nome_arquivo, "rb") as file:
                st.download_button(
                    label="📥 Baixar dados de volatilidade (CSV)",
                    data=file,
                    file_name="dados_volatilidade.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with col2:
            # Converter DataFrame principal para CSV
            csv_bruto = df_ativos.to_csv().encode("utf-8")
            st.download_button(
                label="📥 Baixar dados brutos (CSV)",
                data=csv_bruto,
                file_name="dados_brutos.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"❌ Erro no processamento dos dados: {e}")
        st.info("🔍 Verifique a estrutura do seu arquivo de dados.")
        # Debug information
        with st.expander("🔧 Detalhes do erro para debug"):
            st.write("Tipo de erro:", type(e).__name__)
            if "temp_df" in locals():
                st.write("Colunas no temp_df:", list(temp_df.columns))

else:
    st.info("👈 Configure os dados no menu lateral para começar a análise.")

    # Exemplos de uso
    with st.expander("🎯 Como usar esta aplicação", expanded=True):
        st.markdown(
            """
        **📈 Para usar dados de exemplo:**
        1. Selecione "Usar dados de exemplo"
        2. Digite os tickers (ex: AAPL, MSFT, GOOGL)
        3. Defina a data inicial
        4. Clique em "Carregar Dados de Exemplo"
        
        **📁 Para fazer upload de arquivo:**
        1. Selecione "Fazer upload de arquivo CSV"
        2. Faça upload do seu arquivo
        3. A aplicação detectará automaticamente a estrutura
        
        **📊 Formatos suportados para upload:**
        - **Formato multi-index** (como yfinance):
          ```python
          ('High', 'AAPL'), ('Low', 'AAPL'), ('High', 'MSFT'), ('Low', 'MSFT')
          ```
        - **Formato simples**:
          ```python
          Date, High, Low
          ```
        - **Formato português**:
          ```python
          Data, High, Low
          ```
        """
        )

    with st.expander("📝 Modelo de arquivo CSV para upload"):
        st.write("**Baixe este modelo para testar:**")

        # Criar modelo de dados
        modelo_data = {
            "Date": pd.date_range("2024-01-01", periods=5),
            "High": [150.50, 152.30, 151.80, 153.20, 152.90],
            "Low": [149.20, 150.80, 150.50, 151.60, 151.20],
        }
        modelo_df = pd.DataFrame(modelo_data)

        csv_modelo = modelo_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Baixar modelo CSV",
            data=csv_modelo,
            file_name="modelo_dados_ativos.csv",
            mime="text/csv",
        )

        st.write("**Estrutura do modelo:**")
        st.dataframe(modelo_df)

# Informações adicionais
st.sidebar.markdown("---")
st.sidebar.info(
    """
**💡 Dicas:**
- Use dados de exemplo para testar a aplicação
- Para dados reais, faça upload de arquivos CSV
- Formatos suportados: multi-index ou colunas simples
"""
)
