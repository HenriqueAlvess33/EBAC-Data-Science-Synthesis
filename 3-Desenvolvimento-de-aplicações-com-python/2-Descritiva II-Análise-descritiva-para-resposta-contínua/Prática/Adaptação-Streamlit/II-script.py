import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO


# ------------------------------------------------------
# FUNÇÃO PARA CARREGAR DADOS
# ------------------------------------------------------
@st.cache_data
def load_data(conjunto_de_dados):
    """
    Carrega um arquivo CSV enviado pelo usuário.
    - Retorna um DataFrame pandas.
    - Caso haja erro na leitura, exibe uma mensagem na interface.
    Obs.: o decorador @st.cache_data (mencionado na docstring) não foi aplicado,
    mas poderia ser usado para evitar recarregar o arquivo a cada interação.
    """
    try:
        # Lê o arquivo CSV em um DataFrame e retorna
        # -> Observação: aqui é chamado pd.read_csv, então se o arquivo for .xlsx
        #    esta chamada vai falhar; no seu projeto original isso não é problema.
        return pd.read_csv(conjunto_de_dados)  # Lê o arquivo CSV em um DataFrame
    except Exception as e:
        # Se ocorrer qualquer exceção durante a leitura, exibe a mensagem de erro no app
        st.error(f"Não foi possível carregar o arquivo selecionado. {e}")


# ------------------------------------------------------
# CONFIGURAÇÃO INICIAL DA PÁGINA STREAMLIT
# ------------------------------------------------------
st.set_page_config(
    page_title="Preparando a base",  # Título exibido na aba do navegador
    layout="wide",  # Usa toda a largura da tela
    initial_sidebar_state="expanded",  # Barra lateral aberta por padrão
    page_icon="varig_icon.png",  # Ícone exibido na aba
)


def main():
    titulo_1, titulo_2, titulo_3, titulo_4 = st.columns([2, 6, 1, 1])
    titulo_2.title("Influência das variáveis qualitativas sobre `renda`")

    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Faça o upload do dataset `previsao_renda.csv`",
            type="csv",
            help="Somente serão aceitos arquivos CSV",
        )

    if uploaded_file is not None:
        st.sidebar.success("Dataset carregado")
        previsao_renda = load_data(uploaded_file)

        colunas_existentes = ["Unnamed: 0", "mau", "index"]
        colunas_para_remover = [
            col for col in colunas_existentes if col in previsao_renda.columns
        ]

        previsao_renda.drop(columns=colunas_para_remover, inplace=True)

        colunas_para_selectbox = [
            col for col in previsao_renda.columns if col != "renda"
        ]
        variavel_x = st.selectbox(
            options=colunas_para_selectbox,
            label="Selecione a coluna que irá aparecer no gráfico",
        )

        var_information_1, var_information_2, var_information_3 = st.columns(3)

        with var_information_1:
            st.write("#### Confirmação para variável qualitativa")
            st.metric(
                f"Quantidade de categorias `{variavel_x}`",
                previsao_renda[variavel_x].nunique(),
            )

        with var_information_2:
            st.write("#### Importância de cada categoria")
            st.write("Frequência de cada categoria")
            contagem_de_categorias = previsao_renda[variavel_x].value_counts()
            st.dataframe(pd.DataFrame(contagem_de_categorias))

        with var_information_3:
            resultados = []

            @st.cache_data
            def nulos_cat_check():
                for coluna_agrupadora in previsao_renda.columns:
                    for categoria in (
                        previsao_renda[coluna_agrupadora].value_counts().index
                    ):
                        # Filtra as linhas onde a coluna_agrupadora == categoria
                        df_filtrado = previsao_renda[
                            previsao_renda[coluna_agrupadora] == categoria
                        ]

                        # Para CADA COLUNA do dataframe, calcula a porcentagem de nulos
                        for coluna_analisada in previsao_renda.columns:
                            qtd_nulos = df_filtrado[coluna_analisada].isna().sum()
                            pct_nulos = (qtd_nulos / len(df_filtrado)) * 100

                            resultados.append(
                                {
                                    "Coluna_Agrupadora": coluna_agrupadora,
                                    "Categoria": categoria,
                                    "Coluna_Analisada": coluna_analisada,
                                    "Porcentagem_Nulos": pct_nulos,
                                }
                            )

                df_pct_nulos_por_categoria = pd.DataFrame(resultados)
                df_pct_nulos_por_categoria.sort_values(
                    by="Porcentagem_Nulos", ascending=False, inplace=True
                )
                return df_pct_nulos_por_categoria

            df_pct_nulos_por_categoria = nulos_cat_check()

            st.write("#### Porcentagem de nulos relacionada a cada categoria")
            selecao_cat = st.selectbox(
                options=previsao_renda[variavel_x].value_counts().index,
                label="Selecione a categoria:",
                help="Utilize a categoria escolhida para filtrar o dataframe abaico",
            )
            # Filtra para mostrar apenas quando a coluna agrupadora for a variável selecionada
            st.dataframe(
                df_pct_nulos_por_categoria.loc[
                    (df_pct_nulos_por_categoria["Categoria"] == selecao_cat)
                    & (df_pct_nulos_por_categoria["Coluna_Agrupadora"] == variavel_x)
                ].reset_index(drop=True)
            )
        st.divider()
        st.write("### 📊 Análise gráfica para identificação de possíveis outliers")

        # Opção de seleção de categorias
        opcao_cat = st.radio(
            "Escolha o escopo da análise:",
            ["Utilizar todas as categorias", "Utilizar somente as selecionadas"],
            horizontal=True,
        )

        # Inicializa a variável cat_variaveis
        cat_variaveis = []

        if opcao_cat == "Utilizar somente as selecionadas":
            cat_variaveis = st.multiselect(
                options=previsao_renda[variavel_x].value_counts().index.tolist(),
                label="Selecione as categorias para análise:",
                default=previsao_renda[variavel_x].value_counts().index.tolist()[:3]  # Primeiras 3 por padrão
            )

        def plotagem_boxplot(opcao_cat, cat_variaveis, variavel_x, previsao_renda):
            """
            Função corrigida para plotagem de boxplots
            """
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                if opcao_cat == "Utilizar somente as selecionadas" and cat_variaveis:
                    # CORREÇÃO: Filtrar o DataFrame corretamente
                    dados_filtrados = previsao_renda[previsao_renda[variavel_x].isin(cat_variaveis)]
                    
                    if dados_filtrados.empty:
                        st.warning("⚠️ Nenhum dado encontrado para as categorias selecionadas.")
                        return
                        
                    sns.boxplot(
                        x=variavel_x,
                        y="renda",
                        data=dados_filtrados,
                        ax=ax,
                        palette="viridis"
                    )
                    titulo = f"Boxplot de Renda - Categorias Selecionadas ({len(cat_variaveis)} categorias)"
                else:
                    # Todas as categorias
                    sns.boxplot(
                        x=variavel_x,
                        y="renda",
                        data=previsao_renda,
                        ax=ax,
                        palette="Set2"
                    )
                    titulo = f"Boxplot de Renda - Todas as Categorias"
                
                # Personalização do gráfico
                ax.set_title(titulo, fontsize=14, fontweight='bold')
                ax.set_xlabel(variavel_x, fontsize=12)
                ax.set_ylabel("Renda", fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                
                # Adicionar grid para melhor leitura
                ax.grid(True, alpha=0.3)
                
                # Ajustar layout
                plt.tight_layout()
                
                # Salvar e exibir
                buff = BytesIO()
                fig.savefig(buff, format="png", bbox_inches="tight", dpi=300)
                buff.seek(0)
                st.image(buff, width=800)
                
                # Adicionar estatísticas rápidas
                col1, col2, col3 = st.columns(3)
                with col1:
                    if opcao_cat == "Utilizar somente as selecionadas" and cat_variaveis:
                        st.metric("📊 Categorias Analisadas", len(cat_variaveis))
                    else:
                        st.metric("📊 Categorias Analisadas", previsao_renda[variavel_x].nunique())
                
                with col2:
                    dados_analise = dados_filtrados if (opcao_cat == "Utilizar somente as selecionadas" and cat_variaveis) else previsao_renda
                    st.metric("📈 Renda Média", f"R$ {dados_analise['renda'].mean():.2f}")
                
                with col3:
                    st.metric("🔍 Observações", len(dados_analise))
                    
            except Exception as e:
                st.error(f"❌ Erro ao gerar gráfico: {e}")

        # CORREÇÃO: Chamar a função com todos os parâmetros necessários
        if opcao_cat == "Utilizar somente as selecionadas" and not cat_variaveis:
            st.warning("⚠️ Selecione pelo menos uma categoria para continuar.")
        else:
            plotagem_boxplot(opcao_cat, cat_variaveis, variavel_x, previsao_renda)


# ------------------------------------------------------
# PONTO DE ENTRADA
# ------------------------------------------------------
if __name__ == "__main__":
    main()  # Executa a função principal ao rodar o script

    # informacoes_para_o_grafico = st.radio(
    #     "Categorias a serem apresentadas nos gráficos",
    #     ["Todas", "Selecionadas"],
    #     horizontal=True,
    # )
