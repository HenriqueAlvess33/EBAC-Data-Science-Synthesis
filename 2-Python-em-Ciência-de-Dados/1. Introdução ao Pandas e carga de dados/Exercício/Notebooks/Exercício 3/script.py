import streamlit as st
import pandas as pd
import requests
from typing import Optional, Literal

# Configura o título da aplicação
st.set_page_config(
    page_title="Seleção de dados - Tesouro Nacional",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="varig_icon.png",
)


# Cache para evitar múltiplas requisições desnecessárias
@st.cache_data(show_spinner="Coletando dados do Tesouro Nacional...")
def coleta_dados(
    uf: str, delegacao: Literal["Estado", "Município"]
) -> Optional[pd.DataFrame]:
    """
    Coleta dados da API do Tesouro Nacional baseado na UF e nível territorial.

    Args:
        uf: Sigla do estado (ex: 'SP', 'RJ')
        delegacao: Nível territorial ('Estado' ou 'Município')

    Returns:
        DataFrame com os dados ou None em caso de erro
    """
    uf = uf.strip().upper()

    # Validação básica da UF
    if len(uf) != 2 or not uf.isalpha():
        return None

    endpoints = {
        "Estado": f"https://apidatalake.tesouro.gov.br/ords/sadipem/tt/pvl?uf={uf}&tipo_interessado=Estado",
        "Município": f"https://apidatalake.tesouro.gov.br/ords/sadipem/tt/pvl?uf={uf}&tipo_interessado=Munic%C3%ADpio",
    }

    try:
        _r = requests.get(endpoints[delegacao], timeout=10)
        _r.raise_for_status()  # Levanta exceção para códigos de erro HTTP

        _data_json = _r.json()
        return pd.DataFrame(_data_json["items"])

    except requests.exceptions.RequestException as e:
        st.error(f"Erro na requisição: {e}")
        return None
    except (KeyError, ValueError) as e:
        st.error(f"Erro no processamento dos dados: {e}")
        return None


def main():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">
        <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 2.5em;'>
            <strong>📊 Portal de Dados do Tesouro Nacional</strong>
        </h1>
        <p style='text-align: center; font-family: "Kantumruy Pro", sans-serif;'>
            Consulta de Precatórios por UF e Nível Territorial
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Filtros de Consulta")

        uf = st.text_input(
            "Digite a sigla da UF:",
            placeholder="Ex: SP, RJ, MG",
            max_chars=2,
            help="Digite a sigla do estado com 2 letras",
        ).upper()

        selecao_de_nivel_territorial = st.radio(
            "Nível territorial:",
            ["Estado", "Município"],
            index=0,
            help="Selecione se deseja dados estaduais ou municipais",
        )

        st.info("ℹ️ Os dados são obtidos da API do Tesouro Nacional")

    if uf and len(uf) == 2:
        with st.spinner("Buscando dados..."):
            df = coleta_dados(uf=uf, delegacao=selecao_de_nivel_territorial)

        if df is not None and not df.empty:
            st.success(
                f"✅ Dados encontrados para {uf} - {selecao_de_nivel_territorial}"
            )

            # Estatísticas rápidas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de registros", len(df))
            with col2:
                st.metric("Colunas", len(df.columns))
            with col3:
                st.metric(
                    "Última atualização", pd.to_datetime("today").strftime("%d/%m/%Y")
                )

            st.subheader("📋 Visualização dos Dados")
            st.dataframe(df, use_container_width=True)

            # Expander com metadados
            with st.expander("📝 Metadados e Informações"):
                st.write("**Estrutura dos dados:**")
                st.json(
                    {
                        "Fonte": "Tesouro Nacional - API SADIPEM",
                        "UF": uf,
                        "Nível": selecao_de_nivel_territorial,
                        "Colunas": list(df.columns),
                        "Período": "Dados atualizados regularmente",
                    }
                )

            # Download
            st.subheader("💾 Download dos Dados")
            csv = df.to_csv(index=False, encoding="utf-8-sig")  # utf-8-sig para Excel
            st.download_button(
                label="⬇️ Baixar em CSV",
                data=csv,
                file_name=f"precatorios_{uf}_{selecao_de_nivel_territorial.lower()}_{pd.to_datetime('today').strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download dos dados em formato CSV",
            )

        elif df is not None and df.empty:
            st.warning(
                f"⚠️ Nenhum dado encontrado para {uf} - {selecao_de_nivel_territorial}"
            )
        else:
            st.error("❌ Erro ao consultar os dados. Verifique a UF e tente novamente.")
    elif uf:
        st.warning("⚠️ Por favor, digite uma sigla de UF válida (2 letras)")


if __name__ == "__main__":
    main()
