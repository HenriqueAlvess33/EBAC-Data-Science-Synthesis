import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn import metrics
from scipy.stats import ks_2samp
import statsmodels.formula.api as smf

# Configuração da página
st.set_page_config(page_title="Análise de Modelo Preditivo", layout="wide")
st.title("📊 Análise de Modelo Preditivo - Streamlit")

# Sidebar para upload e configurações
st.sidebar.header("Configurações")

# Upload de arquivo
uploaded_file = st.sidebar.file_uploader(
    "📁 Faça upload do seu arquivo CSV", 
    type=['csv'],
    help="Upload do arquivo de dados"
)

# Se nenhum arquivo for carregado, usar dados padrão
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Arquivo carregado com sucesso!")
else:
    # Carregar dados padrão
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    df = pd.read_csv(url, 
                     names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'])
    st.sidebar.info("Usando dados padrão (Heart Disease)")

# Processamento dos dados
df['flag_doente'] = (df['num']!=0).astype('int64')
df0 = df.copy()

# Sidebar para seleção de variáveis
st.sidebar.header("Seleção de Variáveis")
variavel_alvo = st.sidebar.selectbox(
    "Variável Alvo (Target)",
    options=df.columns,
    index=len(df.columns)-1  # Seleciona a última coluna (flag_doente)
)

# Funções de análise
def analise_bivariada(dataframe, variavel_explicativa, variavel_de_interesse):
    _tab = (
        pd.crosstab(
            dataframe[variavel_explicativa],
            dataframe[variavel_de_interesse],
            margins=True,
        )
        .rename(columns={"All": "Total"}, index={"All": "Total"})
        .assign(Probabilidade=lambda x: round(x[1] / x["Total"], 4) * 100)
        .assign(Odds=lambda x: x[1] / x[0])
        .assign(Odd_ratio_vs_total=lambda x: x["Odds"] / x.loc["Total", "Odds"])
        .reindex(columns=[0, 1, 'Probabilidade',"Odds", "Odd_ratio_vs_total", "Total"])
    )
    return _tab

def analise_bivariada_continua(dataframe, variavel_explicativa, variavel_de_interesse, quantidade_cat=5):
    dataframe[f'cat_{variavel_explicativa}'] = pd.qcut(dataframe[variavel_explicativa], quantidade_cat)
    
    _tab = (
        pd.crosstab(
            dataframe[f'cat_{variavel_explicativa}'],
            dataframe[variavel_de_interesse],
            margins=True,
        )
        .rename(columns={"All": "Total"}, index={"All": "Total"})
        .assign(Probabilidade=lambda x: round(x[1] / x["Total"], 4) * 100)
        .assign(Odds=lambda x: x[1] / x[0])
        .assign(Odd_ratio_vs_total=lambda x: x["Odds"] / x.loc["Total", "Odds"])
        .reindex(columns=[0, 1, 'Probabilidade',"Odds", "Odd_ratio_vs_total", "Total"])
    )
    return _tab

# Layout principal
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Visão Geral", 
    "🔍 Análise Bivariada", 
    "📈 Modelos", 
    "📊 Métricas", 
    "🎯 ROC Curve"
])

with tab1:
    st.header("Visão Geral dos Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Primeiras Linhas")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Informações do Dataset")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Variáveis:** {len(df.columns)}")
        st.write(f"**Registros:** {len(df)}")
        
        buffer = st.container()
        with buffer:
            st.text("Informações de Tipos:")
            buffer_info = st.empty()
            buffer_info.text(str(df.info()))
    
    st.subheader("Estatísticas Descritivas")
    st.dataframe(df.describe())

with tab2:
    st.header("Análise Bivariada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Variáveis Categóricas")
        var_categorica = st.selectbox(
            "Selecione a variável categórica:",
            options=df.select_dtypes(include=['int64', 'object']).columns,
            key="cat_var"
        )
        
        if var_categorica:
            tab_analise = analise_bivariada(df, var_categorica, variavel_alvo)
            st.dataframe(tab_analise)
    
    with col2:
        st.subheader("Variáveis Contínuas")
        var_continua = st.selectbox(
            "Selecione a variável contínua:",
            options=df.select_dtypes(include=['float64', 'int64']).columns,
            key="cont_var"
        )
        
        if var_continua:
            categorias = st.slider("Número de categorias", 3, 10, 5, key="cat_slider")
            tab_analise_cont = analise_bivariada_continua(df, var_continua, variavel_alvo, categorias)
            st.dataframe(tab_analise_cont)

with tab3:
    st.header("Modelos de Regressão Logística")
    
    modelo_selecionado = st.selectbox(
        "Selecione o modelo:",
        [
            "Modelo 1: Variáveis Básicas",
            "Modelo 2: Com Transformações Quadráticas", 
            "Modelo 3: Com Transformações Logarítmicas",
            "Modelo 4: Com Transformações Exponenciais",
            "Modelo 5: Com Variáveis Selecionadas"
        ]
    )
    
    if modelo_selecionado == "Modelo 1: Variáveis Básicas":
        try:
            reglog = smf.logit('flag_doente ~ C(sex) + C(cp) + trestbps + age', data=df).fit()
            df_temp = df.copy()
            df_temp['preditos'] = reglog.predict(df)
            st.success("Modelo 1 ajustado com sucesso!")
        except Exception as e:
            st.error(f"Erro no modelo 1: {e}")
    
    elif modelo_selecionado == "Modelo 2: Com Transformações Quadráticas":
        try:
            reglog = smf.logit('flag_doente ~ C(sex) + C(cp) + np.power(trestbps, 2) + np.power(age, 2)', data=df).fit()
            df_temp = df.copy()
            df_temp['preditos'] = reglog.predict(df)
            st.success("Modelo 2 ajustado com sucesso!")
        except Exception as e:
            st.error(f"Erro no modelo 2: {e}")
    
    elif modelo_selecionado == "Modelo 3: Com Transformações Logarítmicas":
        try:
            reglog = smf.logit('flag_doente ~ C(sex) + C(cp) + np.log(trestbps + .1) + np.log(age + .1)', data=df).fit()
            df_temp = df.copy()
            df_temp['preditos'] = reglog.predict(df)
            st.success("Modelo 3 ajustado com sucesso!")
        except Exception as e:
            st.error(f"Erro no modelo 3: {e}")
    
    elif modelo_selecionado == "Modelo 4: Com Transformações Exponenciais":
        try:
            reglog = smf.logit('flag_doente ~ C(sex) + C(cp) + np.exp(trestbps) + np.exp(age)', data=df).fit()
            df_temp = df.copy()
            df_temp['preditos'] = reglog.predict(df)
            st.success("Modelo 4 ajustado com sucesso!")
        except Exception as e:
            st.error(f"Erro no modelo 4: {e}")
    
    elif modelo_selecionado == "Modelo 5: Com Variáveis Selecionadas":
        try:
            reglog = smf.logit('flag_doente ~ C(sex) + C(cp) + C(thal) + ca + age', data=df).fit()
            df_temp = df.copy()
            df_temp['preditos'] = reglog.predict(df)
            st.success("Modelo 5 ajustado com sucesso!")
        except Exception as e:
            st.error(f"Erro no modelo 5: {e}")
    
    # Mostrar resumo do modelo
    if 'reglog' in locals():
        st.subheader("Resumo do Modelo")
        st.text(str(reglog.summary()))

with tab4:
    st.header("Métricas de Performance")
    
    if 'df_temp' in locals() and 'preditos' in df_temp.columns:
        # Cálculo das métricas
        acc = metrics.accuracy_score(df_temp['flag_doente'], df_temp['preditos'] > 0.5)
        fpr, tpr, thresholds = metrics.roc_curve(df_temp['flag_doente'], df_temp['preditos'])
        auc_ = metrics.auc(fpr, tpr)
        gini = 2 * auc_ - 1
        ks = ks_2samp(
            df_temp.loc[df_temp['flag_doente'] == 1, 'preditos'], 
            df_temp.loc[df_temp['flag_doente'] != 1, 'preditos']
        ).statistic
        
        # Exibição das métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Acurácia", f"{acc*100:.2f}%")
        
        with col2:
            st.metric("KS", f"{ks*100:.2f}%")
        
        with col3:
            st.metric("AUC", f"{auc_*100:.2f}%")
        
        with col4:
            st.metric("GINI", f"{gini*100:.2f}%")
        
        # Análise por grupos de predição
        st.subheader("Análise por Grupos de Predição")
        analise_grupos = analise_bivariada_continua(df_temp, 'preditos', 'flag_doente')
        st.dataframe(analise_grupos)
        
    else:
        st.warning("Execute um modelo na aba 'Modelos' primeiro.")

with tab5:
    st.header("Curva ROC e Análise Gráfica")
    
    if 'df_temp' in locals() and 'preditos' in df_temp.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Curva ROC
            st.subheader("Curva ROC")
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            
            fpr, tpr, thresholds = metrics.roc_curve(df_temp['flag_doente'], df_temp['preditos'])
            auc_ = metrics.auc(fpr, tpr)
            
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (area = {auc_:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic')
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, alpha=0.3)
            
            st.pyplot(fig_roc)
        
        with col2:
            # Gráfico de qualidade
            st.subheader("Qualidade do Modelo")
            df_temp['cat_preditos'] = pd.qcut(df_temp['preditos'], 5)
            group_reg = df_temp.groupby("cat_preditos", observed=False)
            qualid = (
                group_reg[["flag_doente"]]
                .count()
                .rename(columns={"flag_doente": "contagem"})
                .assign(média_preditos=group_reg[["preditos"]].mean())
                .assign(média_doentes=group_reg["flag_doente"].mean())
            )
            
            fig_qual, ax_qual = plt.subplots(figsize=(8, 6))
            ax_qual.plot(qualid['média_doentes'].values, marker='o', label='% Observado', linewidth=2)
            ax_qual.plot(qualid['média_preditos'].values, marker='s', label='% Predito', linewidth=2)
            ax_qual.legend(loc="lower right")
            ax_qual.set_ylabel('Probabilidade de evento')
            ax_qual.set_xlabel('Grupo')
            ax_qual.set_xticks(range(5))
            ax_qual.set_xticklabels([1, 2, 3, 4, 5])
            ax_qual.grid(True, alpha=0.3)
            ax_qual.set_title('Comparação: Observado vs Predito')
            
            st.pyplot(fig_qual)
    
    else:
        st.warning("Execute um modelo na aba 'Modelos' primeiro.")

# Informações no sidebar
st.sidebar.header("Informações")
st.sidebar.info("""
Esta aplicação realiza análise preditiva usando regressão logística. 
Faça upload de seus dados ou use o dataset padrão.
""")

# Rodapé
st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido com Streamlit")

