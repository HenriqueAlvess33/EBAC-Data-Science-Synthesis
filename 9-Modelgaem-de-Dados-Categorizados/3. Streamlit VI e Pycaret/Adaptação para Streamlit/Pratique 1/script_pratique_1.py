import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t, ks_2samp
from sklearn import metrics
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from patsy import dmatrix
import io
import pickle
import json
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Credit Scoring Analysis", layout="wide")

# Título da aplicação
st.title("📊 Sistema de Credit Scoring")
st.markdown("""
Esta aplicação realiza análise de credit scoring para cartão de crédito utilizando 
15 safras de dados e 12 meses de performance.
""")

# Sidebar para upload de arquivos
st.sidebar.header("📁 Upload de Dados")
uploaded_file = st.sidebar.file_uploader(
    "Carregue o arquivo credit_scoring.ftr", 
    type=['ftr'],
    help="Selecione o arquivo feather contendo os dados de credit scoring"
)

# Variáveis globais para armazenar modelo e dados
if 'modelo_treinado' not in st.session_state:
    st.session_state.modelo_treinado = None
if 'df_treino_processado' not in st.session_state:
    st.session_state.df_treino_processado = None
if 'df_teste_processado' not in st.session_state:
    st.session_state.df_teste_processado = None
if 'dict_bins' not in st.session_state:
    st.session_state.dict_bins = None

# Funções auxiliares (mantidas do código original)
def atualizar_metadados(df):
    metadados = pd.DataFrame(df.dtypes, columns=['dtype'])
    metadados['n_missings'] = df.isna().sum()
    metadados['Valores únicos'] = df.nunique()
    metadados = metadados.drop(['data_ref', 'index'], axis=0, errors='ignore')
    return metadados

def poder_de_predicao(df, modelo):
    df2 = df.copy()
    df2['score'] = modelo.predict(df2)
    acc = metrics.accuracy_score(df2.mau, df2.score > .068)
    fpr, tpr, thresholds = metrics.roc_curve(df2.mau, df2.score)
    auc = metrics.auc(fpr, tpr)
    gini = 2 * auc - 1
    ks = ks_2samp(df2.loc[df2.mau == 1, 'score'], df2.loc[df2.mau != 1, 'score']).statistic
    
    return {
        'Acurácia': f'{acc:.1%}',
        'AUC': f'{auc:.1%}',
        'GINI': f'{gini:.1%}',
        'KS': f'{ks:.1%}'
    }

def vif_filter(X, limite=10):
    X_filtrado = X.copy()
    removed_features = []
    vif_scores = {}

    while True:
        X_with_const = add_constant(X_filtrado)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_with_const.columns
        vif_data["VIF"] = [vif(X_with_const.values, i) for i in range(X_with_const.shape[1])]
        vif_data = vif_data[vif_data['feature'] != 'const']

        for _, row in vif_data.iterrows():
            vif_scores[row['feature']] = row['VIF']

        max_vif = vif_data['VIF'].max()
        if max_vif <= limite:
            break

        feature_to_remove = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
        removed_features.append((feature_to_remove, max_vif))
        X_filtrado = X_filtrado.drop(columns=[feature_to_remove])

        if X_filtrado.shape[1] == 0:
            raise ValueError("Todas as variáveis foram removidas - limite pode estar muito baixo")

    remaining_vif = vif_data[vif_data['feature'].isin(X_filtrado.columns)]
    remaining_vif = remaining_vif.sort_values('VIF', ascending=False)

    return X_filtrado, removed_features, remaining_vif

def IV(variavel, resposta):
    tab = pd.crosstab(variavel, resposta, margins=True, margins_name='total')
    rótulo_evento = 1
    rótulo_nao_evento = 0
    tab['pct_evento'] = tab[rótulo_evento]/tab.loc['total',rótulo_evento]
    tab['pct_nao_evento'] = tab[rótulo_nao_evento]/tab.loc['total',rótulo_nao_evento]
    tab['woe'] = np.log(tab.pct_evento/tab.pct_nao_evento)
    tab['iv_parcial'] = (tab.pct_evento - tab.pct_nao_evento)*tab.woe
    return tab['iv_parcial'].sum()

# Funções para download
def converter_df_para_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def salvar_modelo(modelo, dict_bins, metadados):
    modelo_data = {
        'modelo': modelo,
        'dict_bins': dict_bins,
        'metadados': metadados,
        'data_treinamento': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'versao': '1.0'
    }
    return pickle.dumps(modelo_data)

# Processamento principal
if uploaded_file is not None:
    try:
        # Carregar dados
        df_original = pd.read_feather(uploaded_file)
        
        # Abas principais
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Visão Geral", 
            "🔍 Análise Descritiva", 
            "🤖 Modelagem", 
            "📊 Resultados",
            "💾 Download"
        ])

        with tab1:
            st.header("Visão Geral dos Dados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Metadados")
                metadados = atualizar_metadados(df_original)
                st.dataframe(metadados, use_container_width=True)
            
            with col2:
                st.subheader("Estatísticas Básicas")
                st.dataframe(df_original.describe(), use_container_width=True)
            
            st.subheader("Primeiras Linhas dos Dados")
            st.dataframe(df_original.head(), use_container_width=True)

        with tab2:
            st.header("Análise Descritiva")
            
            # Preparação dos dados
            df_trabalho = df_original.loc[df_original['data_ref'].dt.year == 2015]
            df_trabalho['mau'] = df_trabalho['mau'].astype('int')
            
            df_oot = df_original.loc[df_original['data_ref'].dt.year == 2016]
            df_oot['mau'] = df_oot['mau'].astype('int')
            
            # Amostragem
            df_trabalho['Mês'] = df_trabalho['data_ref'].dt.month
            amostras = []
            
            for mes in df_trabalho['Mês'].unique().tolist():
                amostra_mes = df_trabalho.loc[df_trabalho['Mês'] == mes].sample(2000, random_state=42)
                amostras.append(amostra_mes)
            
            df_treino = pd.concat(amostras)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribuição por Tipo de Renda e Sexo")
                fig, ax = plt.subplots()
                sns.countplot(x='tipo_renda', data=df_treino, hue='sexo', ax=ax)
                plt.xticks(rotation=45)
                ax.set_title("Distribuição por tipo de renda e sexo")
                ax.set_xlabel("Tipo de Renda")
                ax.set_ylabel("Quantidade")
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.subheader("Quantidade de Filhos por Tipo de Renda")
                fig, ax = plt.subplots()
                sns.barplot(data=df_treino, x='tipo_renda', y='qtd_filhos', hue='sexo')
                plt.xticks(rotation=45)
                ax.set_title("Quantidade de filhos por tipo de renda")
                ax.set_xlabel("Tipo de Renda")
                ax.set_ylabel("Quantidade de filhos")
                plt.tight_layout()
                st.pyplot(fig)
            
            st.subheader("Distribuição de Renda por Tipo de Renda")
            fig, ax = plt.subplots()
            sns.boxplot(data=df_treino, x='tipo_renda', y='renda', hue='sexo')
            plt.xticks(rotation=45)
            ax.set_title("Distribuição de renda por tipo de renda")
            ax.set_xlabel("Tipo de Renda")
            ax.set_ylabel("Renda")
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            st.header("Desenvolvimento do Modelo")
            
            # Tratamento de missings
            df_treino['tempo_emprego_missing'] = (df_treino['tempo_emprego'].isna() == 1).astype(int)
            
            # Engenharia de features
            dict_bins = {}
            df_treino_1 = df_treino.copy()
            
            # Criação de categorias para variáveis contínuas
            _, cat_tempo_emprego_bins = pd.qcut(
                df_treino_1["tempo_emprego"], q=20, duplicates="drop", precision=0, retbins=True
            )
            
            df_treino_1["cat_tempo_emprego"] = pd.cut(
                df_treino_1["tempo_emprego"],
                bins=cat_tempo_emprego_bins,
                precision=0,
                duplicates="drop",
            )
            df_treino_1["cat_tempo_emprego"] = (
                df_treino_1["cat_tempo_emprego"].cat.add_categories("Missing").fillna("Missing")
            )
            
            dict_bins['tempo_emprego'] = cat_tempo_emprego_bins
            
            for col in ["qtd_filhos", "qt_pessoas_residencia", "renda"]:
                _, bins = pd.qcut(df_treino_1[col], 20, retbins=True, precision=0, duplicates="drop")
                cat = pd.cut(df_treino_1[col], bins=bins, precision=0, duplicates="drop")
                df_treino_1[f"cat_{col}"] = cat.cat.add_categories("Missing").fillna("Missing")
                dict_bins[col] = bins
            
            df_treino_1.drop(
                columns=["renda", "qt_pessoas_residencia", "qtd_filhos", "tempo_emprego"],
                inplace=True,
            )
            
            # Cálculo do IV
            metadados_02 = atualizar_metadados(df_treino_1)
            iv_dict = {}
            
            for variavel in metadados_02.index.to_list():
                if variavel not in [
                    "tempo_emprego", "qtd_filhos", "qt_pessoas_residencia", 
                    "renda", "mau"
                ]:
                    iv_calculado = IV(df_treino_1[variavel], df_treino_1["mau"])
                    iv_dict[variavel] = iv_calculado
            
            metadados_02["IV"] = metadados_02.index.map(iv_dict)
            iv_aprovado = metadados_02.loc[metadados_02["IV"] >= 0.02].index.to_list()
            
            st.subheader("Variáveis Selecionadas (IV ≥ 0.02)")
            st.dataframe(metadados_02[metadados_02["IV"] >= 0.02][['dtype', 'n_missings', 'Valores únicos', 'IV']])
            
            # Modelagem
            if st.button("Treinar Modelo de Regressão Logística"):
                with st.spinner("Treinando modelo..."):
                    formula = '+'.join([x for x in iv_aprovado if x not in [
                        'mau', 'Ano', 'Ano_Trimestre', 'Dia do Mês', 
                        'Dia_semana', 'Nome_mes', 'Mês'
                    ]])
                    
                    modelo = smf.logit(f'mau ~ {formula}', df_treino_1).fit()
                    
                    # Armazenar modelo e dados na session state
                    st.session_state.modelo_treinado = modelo
                    st.session_state.df_treino_processado = df_treino_1
                    st.session_state.dict_bins = dict_bins
                    
                    st.success("Modelo treinado com sucesso!")
                    st.subheader("Resumo do Modelo")
                    st.text(str(modelo.summary()))

        with tab4:
            st.header("Avaliação do Modelo")
            
            if st.session_state.modelo_treinado is not None:
                modelo = st.session_state.modelo_treinado
                df_treino_1 = st.session_state.df_treino_processado
                dict_bins = st.session_state.dict_bins
                
                # Preparação da base OOT
                df_teste = df_oot.copy()
                
                for col in ["tempo_emprego", "qtd_filhos", "qt_pessoas_residencia", "renda"]:
                    cat = pd.cut(df_teste[col], bins=dict_bins[f'{col}'], precision=0, duplicates="drop")
                    df_teste[f'cat_{col}'] = cat
                    df_teste[f"cat_{col}"] = cat.cat.add_categories("Missing").fillna("Missing")
                
                df_teste.drop(
                    columns=['renda', 'qt_pessoas_residencia', 'qtd_filhos', 'tempo_emprego'], 
                    inplace=True
                )
                
                # Armazenar dados de teste processados
                st.session_state.df_teste_processado = df_teste
                
                # Avaliação
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Base de Treino")
                    metrics_treino = poder_de_predicao(df_treino_1, modelo)
                    for metric, value in metrics_treino.items():
                        st.metric(metric, value)
                
                with col2:
                    st.subheader("Base de Teste (OOT)")
                    metrics_teste = poder_de_predicao(df_teste, modelo)
                    for metric, value in metrics_teste.items():
                        st.metric(metric, value)
                
                # Curva ROC
                st.subheader("Curva ROC")
                fig, ax = plt.subplots()
                
                for df, label in [(df_treino_1, 'Treino'), (df_teste, 'Teste')]:
                    df_temp = df.copy()
                    df_temp['score'] = modelo.predict(df_temp)
                    fpr, tpr, _ = metrics.roc_curve(df_temp.mau, df_temp.score)
                    auc = metrics.auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})')
                
                ax.plot([0, 1], [0, 1], 'k--', label='Linha de Referência')
                ax.set_xlabel('Taxa de Falsos Positivos')
                ax.set_ylabel('Taxa de Verdadeiros Positivos')
                ax.set_title('Curva ROC')
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Treine o modelo primeiro na aba 'Modelagem' para ver os resultados.")

        with tab5:
            st.header("💾 Download de Arquivos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Download do Dataset")
                
                # Dataset original
                csv_original = converter_df_para_csv(df_original)
                st.download_button(
                    label="📥 Download Dataset Original (CSV)",
                    data=csv_original,
                    file_name="credit_scoring_dataset_original.csv",
                    mime="text/csv",
                    help="Baixe o dataset original completo em formato CSV"
                )
                
                # Dataset de treino processado
                if st.session_state.df_treino_processado is not None:
                    csv_treino = converter_df_para_csv(st.session_state.df_treino_processado)
                    st.download_button(
                        label="📥 Download Dataset Treino Processado (CSV)",
                        data=csv_treino,
                        file_name="credit_scoring_treino_processado.csv",
                        mime="text/csv",
                        help="Baixe o dataset de treino após processamento e feature engineering"
                    )
                
                # Dataset de teste processado
                if st.session_state.df_teste_processado is not None:
                    csv_teste = converter_df_para_csv(st.session_state.df_teste_processado)
                    st.download_button(
                        label="📥 Download Dataset Teste Processado (CSV)",
                        data=csv_teste,
                        file_name="credit_scoring_teste_processado.csv",
                        mime="text/csv",
                        help="Baixe o dataset de teste (OOT) após processamento"
                    )
            
            with col2:
                st.subheader("Download do Modelo")
                
                if st.session_state.modelo_treinado is not None:
                    # Preparar dados do modelo para download
                    modelo_data = salvar_modelo(
                        st.session_state.modelo_treinado,
                        st.session_state.dict_bins,
                        metadados_02
                    )
                    
                    st.download_button(
                        label="🤖 Download Modelo Treinado (PKL)",
                        data=modelo_data,
                        file_name=f"credit_scoring_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl",
                        mime="application/octet-stream",
                        help="Baixe o modelo treinado com todos os parâmetros e bins"
                    )
                    
                    # Download dos parâmetros do modelo em JSON
                    params_json = json.dumps({
                        'data_treinamento': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'n_observacoes_treino': len(st.session_state.df_treino_processado),
                        'variaveis_utilizadas': iv_aprovado,
                        'metricas_treino': poder_de_predicao(st.session_state.df_treino_processado, st.session_state.modelo_treinado)
                    }, indent=2)
                    
                    st.download_button(
                        label="📋 Download Metadados do Modelo (JSON)",
                        data=params_json,
                        file_name="modelo_metadados.json",
                        mime="application/json",
                        help="Baixe os metadados e parâmetros do modelo"
                    )
                else:
                    st.info("Treine o modelo primeiro para habilitar o download.")
            
            st.subheader("📊 Relatórios e Análises")
            
            # Relatório de métricas
            if st.session_state.modelo_treinado is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Relatório de performance
                    relatorio_performance = f"""
                    RELATÓRIO DE PERFORMANCE DO MODELO
                    Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    
                    MÉTRICAS DE TREINO:
                    {chr(10).join([f'{k}: {v}' for k, v in poder_de_predicao(st.session_state.df_treino_processado, st.session_state.modelo_treinado).items()])}
                    
                    MÉTRICAS DE TESTE (OOT):
                    {chr(10).join([f'{k}: {v}' for k, v in poder_de_predicao(st.session_state.df_teste_processado, st.session_state.modelo_treinado).items()])}
                    
                    VARIÁVEIS UTILIZADAS: {len(iv_aprovado)}
                    """
                    
                    st.download_button(
                        label="📄 Download Relatório de Performance (TXT)",
                        data=relatorio_performance,
                        file_name="relatorio_performance_modelo.txt",
                        mime="text/plain",
                        help="Baixe um relatório resumido com as métricas de performance do modelo"
                    )
                
                with col2:
                    # Download dos bins utilizados
                    bins_json = json.dumps(
                        {k: v.tolist() if hasattr(v, 'tolist') else str(v) for k, v in st.session_state.dict_bins.items()},
                        indent=2
                    )
                    
                    st.download_button(
                        label="📐 Download Bins das Variáveis (JSON)",
                        data=bins_json,
                        file_name="variaveis_bins.json",
                        mime="application/json",
                        help="Baixe os pontos de corte utilizados para discretização das variáveis"
                    )

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {str(e)}")
else:
    st.info("👈 Por favor, faça o upload do arquivo 'credit_scoring.ftr' na barra lateral para iniciar a análise.")
    
    # Exemplo de estrutura esperada
    st.subheader("Estrutura Esperada do Arquivo")
    st.markdown("""
    O arquivo deve conter as seguintes colunas principais:
    - `data_ref`: Data de referência
    - `mau`: Indicador de mau pagador (target)
    - `sexo`, `posse_de_veiculo`, `posse_de_imovel`: Dados demográficos
    - `tipo_renda`, `educacao`, `estado_civil`: Dados socioeconômicos
    - `qtd_filhos`, `qt_pessoas_residencia`: Dados familiares
    - `idade`, `tempo_emprego`, `renda`: Dados numéricos
    """)

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido para análise de credit scoring | Streamlit App")