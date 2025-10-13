import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import io

# Configuração da página
st.set_page_config(page_title="Classificação de Atividade Humana com PCA", layout="wide")

st.title("Classificação de Atividade Humana com PCA")

# Definir funções primeiro
def padroniza(s):
    if s.std() > 0:
        s = (s - s.mean())/s.std()
    return s

# Função para plotar o scree plot
def screeplot(princomp, ncomp=0, varexplicada=0, criterio=1):
    if ncomp > 0:
        ncomp_crit = ncomp
    elif varexplicada > 0:
        ncomp_crit = (
            princomp.explained_variance_ratio_.cumsum() < varexplicada
        ).sum() + 1
    elif criterio == 1:
        ncomp_crit = (
            princomp.explained_variance_ratio_ > 1 / princomp.n_components_
        ).sum()
    else:
        ncomp_crit = None

    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(14, 8))
    plt.subplots_adjust(hspace=0, wspace=0.15)

    num_componentes = np.arange(princomp.n_components_) + 1

    # Gráfico da variância explicada por componente
    ax[0, 0].plot(
        num_componentes,
        princomp.explained_variance_,
        "o-",
        linewidth=2,
        color="blue",
        markersize=2,
        alpha=0.2,
    )
    ax[0, 0].set_title("Scree Plot - Variância total")
    ax[0, 0].set_xlabel("Número de componentes")
    ax[0, 0].set_ylabel("Variancia explicada (Autovalores)")

    # Gráfico da variância explicada acumulada
    ax[1, 0].plot(
        num_componentes,
        princomp.explained_variance_.cumsum(),
        "o-",
        linewidth=2,
        color="blue",
        markersize=2,
        alpha=0.2,
    )
    ax[1, 0].set_xlabel("Número de componentes")
    ax[1, 0].set_ylabel("Variancia explicada (Acumulada)")

    # Gráfico da variância percentual explicada por componente
    ax[0, 1].plot(
        num_componentes,
        princomp.explained_variance_ratio_,
        "o-",
        linewidth=2,
        color="blue",
        markersize=2,
        alpha=0.2,
    )
    ax[0, 1].set_title("Scree Plot - Variância percentual")
    ax[0, 1].set_xlabel("Número de componentes")
    ax[0, 1].set_ylabel("Variancia explicada (percentual)")

    # Gráfico da variância percentual acumulada
    ax[1, 1].plot(
        num_componentes,
        princomp.explained_variance_ratio_.cumsum(),
        "o-",
        linewidth=2,
        color="blue",
        markersize=2,
        alpha=0.2,
    )
    ax[1, 1].set_xlabel("Número de componentes")
    ax[1, 1].set_ylabel("Variancia explicada (% Acumulado)")

    if ncomp_crit is not None:
        # Linhas verticais de referência
        for i in range(2):
            for j in range(2):
                ax[i, j].axvline(x=ncomp_crit, color="r", linestyle="-", linewidth=0.5)

        # Linhas horizontais de referência
        variancia = princomp.explained_variance_[ncomp_crit - 1]
        variancia_acumulada = princomp.explained_variance_.cumsum()[ncomp_crit - 1]
        pct_variancia = princomp.explained_variance_ratio_[ncomp_crit - 1]
        pct_variancia_acumulada = princomp.explained_variance_ratio_.cumsum()[ncomp_crit - 1]

        ax[0, 0].axhline(y=variancia, color="r", linestyle="-", linewidth=0.5)
        ax[1, 0].axhline(y=variancia_acumulada, color="r", linestyle="-", linewidth=0.5)
        ax[0, 1].axhline(y=pct_variancia, color="r", linestyle="-", linewidth=0.5)
        ax[1, 1].axhline(y=pct_variancia_acumulada, color="r", linestyle="-", linewidth=0.5)

    return fig, ncomp_crit, variancia, variancia_acumulada, pct_variancia, pct_variancia_acumulada

# Área para upload dos arquivos
st.sidebar.header("Upload dos Arquivos")

# Função para carregar arquivos
def load_file(uploaded_file, **kwargs):
    if uploaded_file is not None:
        return pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")), **kwargs)
    return None

# Upload dos arquivos necessários
uploaded_features = st.sidebar.file_uploader("Arquivo features.txt", type=["txt"])
uploaded_labels = st.sidebar.file_uploader("Arquivo activity_labels.txt", type=["txt"])
uploaded_subtrain = st.sidebar.file_uploader("Arquivo subject_train.txt", type=["txt"])
uploaded_xtrain = st.sidebar.file_uploader("Arquivo X_train.txt", type=["txt"])
uploaded_ytrain = st.sidebar.file_uploader("Arquivo y_train.txt", type=["txt"])
uploaded_subtest = st.sidebar.file_uploader("Arquivo subject_test.txt", type=["txt"])
uploaded_xtest = st.sidebar.file_uploader("Arquivo X_test.txt", type=["txt"])
uploaded_ytest = st.sidebar.file_uploader("Arquivo y_test.txt", type=["txt"])

# Verifica se todos os arquivos foram carregados
all_files_uploaded = all([uploaded_features, uploaded_labels, uploaded_subtrain, 
                         uploaded_xtrain, uploaded_ytrain, uploaded_subtest, 
                         uploaded_xtest, uploaded_ytest])

if not all_files_uploaded:
    st.warning("Por favor, faça upload de todos os arquivos necessários para continuar.")
    st.stop()

# Carregando os dados
try:
    # Carregando os nomes das variáveis (features)
    features = pd.read_csv(
        uploaded_features, header=None, names=["nome_var"], sep="#"
    ).squeeze()

    # Carregando os rótulos das atividades
    labels = pd.read_csv(
        uploaded_labels, delim_whitespace=True, header=None, names=["cod_label", "label"]
    )

    # Carregando os IDs dos sujeitos no conjunto de treino
    subject_train = pd.read_csv(
        uploaded_subtrain, header=None, names=["subject_id"]
    ).squeeze()

    # Carregando os dados de treino
    X_train = pd.read_csv(
        uploaded_xtrain, delim_whitespace=True, header=None, names=features.tolist()
    )
    y_train = pd.read_csv(
        uploaded_ytrain, header=None, names=["cod_label"]
    )

    # Carregando os IDs dos sujeitos no conjunto de teste
    subject_test = pd.read_csv(
        uploaded_subtest, header=None, names=["subject_id"]
    ).squeeze()

    # Carregando os dados de teste
    X_test = pd.read_csv(
        uploaded_xtest, delim_whitespace=True, header=None, names=features.tolist()
    )
    y_test = pd.read_csv(
        uploaded_ytest, header=None, names=["cod_label"]
    )

    st.success("Todos os arquivos foram carregados com sucesso!")

    # Informações sobre os dados
    with st.sidebar:
        st.header("Informações dos Dados")
        st.write(f"**Formato X_train:** {X_train.shape}")
        st.write(f"**Formato X_test:** {X_test.shape}")
        st.write(f"**Número de features:** {len(features)}")
        st.write(f"**Número de atividades:** {len(labels)}")

except Exception as e:
    st.error(f"Erro ao carregar os arquivos: {e}")
    st.stop()

# PCA sem padronização
st.header("PCA sem Padronização")

if st.button("Executar PCA sem Padronização"):
    with st.spinner("Executando PCA..."):
        pca = PCA()
        princomp = pca.fit(X_train)
        comp1_train = pca.transform(X_train)
        comp1_test = pca.transform(X_test)

        # Criando DataFrame com resultados
        df_pca = (
            pd.DataFrame(princomp.explained_variance_)
            .reset_index()
            .rename(columns={"index": "Componentes", 0: "Variância explicada"})
            .merge(
                pd.DataFrame(princomp.explained_variance_ratio_ * 100)
                .reset_index()
                .rename(columns={"index": "Componentes", 0: "Proporção Variância explicada"})
            )
            .merge(
                pd.DataFrame(princomp.explained_variance_ratio_.cumsum() * 100)
                .reset_index()
                .rename(columns={"index": "Componentes", 0: "Proporção Variância acumulada"})
            )
        )

        st.subheader("Resultados do PCA sem Padronização")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Soma da variância explicada", round(sum(princomp.explained_variance_), 3))
        with col2:
            st.metric("Soma da variância percentual", round(sum(princomp.explained_variance_ratio_), 3))
        with col3:
            st.metric("Número de componentes", len(princomp.explained_variance_ratio_))

        # Scree plot
        st.subheader("Scree Plot - PCA sem Padronização")
        fig, ncomp_crit, variancia, variancia_acumulada, pct_variancia, pct_variancia_acumulada = screeplot(princomp, varexplicada=0.90)
        st.pyplot(fig)

        # Informações sobre os componentes
        st.subheader("Informações dos Componentes Principais")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Número de componentes para 90% de variância:** {ncomp_crit}")
            st.write(f"**Variância da última CP:** {variancia:.2f}")
            st.write(f"**Variância total explicada:** {variancia_acumulada:.2f}")
        with col2:
            st.write(f"**Variância percentual da última CP:** {100*pct_variancia:.2f}%")
            st.write(f"**Variância percentual total explicada:** {100*pct_variancia_acumulada:.2f}%")

        # Tabela com os primeiros componentes
        st.subheader("Primeiros Componentes Principais")
        st.dataframe(df_pca.head(10))

# PCA com padronização
st.header("PCA com Padronização")

if st.button("Executar PCA com Padronização"):
    with st.spinner("Executando PCA com padronização..."):
        X_train_pad = pd.DataFrame(X_train).apply(padroniza, axis=0)
        X_test_pad = pd.DataFrame(X_test).apply(padroniza, axis=0)

        pca = PCA()
        prcomp2 = pca.fit(X_train_pad)
        comp2_train = pca.transform(X_train_pad)
        comp2_test = pca.transform(X_test_pad)

        # Criando DataFrame com resultados
        df_pca_pad = (
            pd.DataFrame(prcomp2.explained_variance_)
            .reset_index()
            .rename(columns={"index": "Componentes", 0: "Variância explicada"})
            .merge(
                pd.DataFrame(prcomp2.explained_variance_ratio_ * 100)
                .reset_index()
                .rename(columns={"index": "Componentes", 0: "Proporção Variância explicada"})
            )
            .merge(
                pd.DataFrame(prcomp2.explained_variance_ratio_.cumsum() * 100)
                .reset_index()
                .rename(columns={"index": "Componentes", 0: "Proporção Variância acumulada"})
            )
        )

        st.subheader("Resultados do PCA com Padronização")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Soma da variância explicada", round(sum(prcomp2.explained_variance_), 3))
        with col2:
            st.metric("Soma da variância percentual", round(sum(prcomp2.explained_variance_ratio_), 3))
        with col3:
            st.metric("Número de componentes", len(prcomp2.explained_variance_ratio_))

        # Scree plot
        st.subheader("Scree Plot - PCA com Padronização")
        fig, ncomp_crit, variancia, variancia_acumulada, pct_variancia, pct_variancia_acumulada = screeplot(prcomp2, varexplicada=0.90)
        st.pyplot(fig)

        # Informações sobre os componentes
        st.subheader("Informações dos Componentes Principais")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Número de componentes para 90% de variância:** {ncomp_crit}")
            st.write(f"**Variância da última CP:** {variancia:.2f}")
            st.write(f"**Variância total explicada:** {variancia_acumulada:.2f}")
        with col2:
            st.write(f"**Variância percentual da última CP:** {100*pct_variancia:.2f}%")
            st.write(f"**Variância percentual total explicada:** {100*pct_variancia_acumulada:.2f}%")

        # Tabela com os primeiros componentes
        st.subheader("Primeiros Componentes Principais")
        st.dataframe(df_pca_pad.head(10))

# Classificação com Árvore de Decisão
st.header("Classificação com Árvore de Decisão e PCA")

if st.button("Executar Classificação"):
    with st.spinner("Treinando modelos..."):
        # PCA para 10 componentes
        pca = PCA(n_components=10)
        
        # Sem padronização
        princomp = pca.fit(X_train)
        comp1_train = pca.transform(X_train)
        comp1_test = pca.transform(X_test)
        
        # Com padronização
        X_train_pad = pd.DataFrame(X_train).apply(padroniza, axis=0)
        X_test_pad = pd.DataFrame(X_test).apply(padroniza, axis=0)
        prcomp2 = pca.fit(X_train_pad)
        comp2_train = pca.transform(X_train_pad)
        comp2_test = pca.transform(X_test_pad)

        # Treinando modelos
        clf_nd = DecisionTreeClassifier(ccp_alpha=0.001)
        clf_nd = clf_nd.fit(comp1_train, y_train)

        clf_def = DecisionTreeClassifier(ccp_alpha=0.001)
        clf_def = clf_def.fit(comp2_train, y_train)

        # Avaliando modelos
        acuracia_dos_modelos = {
            'acuracia_treino': {
                "sem_padronizacao": [],
                "com_padronizacao": [],
            },
            'acuracia_teste': {
                "sem_padronizacao": [],
                "com_padronizacao": [],
            }
        }

        # Modelo sem padronização
        st.subheader("Modelo sem Padronização")
        col1, col2 = st.columns(2)
        
        with col1:
            y_pred_test = clf_nd.predict(comp1_test)
            y_pred_train = clf_nd.predict(comp1_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            
            acuracia_dos_modelos['acuracia_teste']['sem_padronizacao'].append(acc_test)
            acuracia_dos_modelos['acuracia_treino']['sem_padronizacao'].append(acc_train)
            
            st.metric("Acurácia Treino", f"{acc_train:.3f}")
            st.metric("Acurácia Teste", f"{acc_test:.3f}")
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_estimator(
                clf_nd,
                comp1_test,
                y_test,
                display_labels=labels.label.tolist(),
                cmap=plt.cm.Blues,
                normalize="true",
                ax=ax
            )
            plt.title("Matriz de Confusão - Sem Padronização")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

        # Modelo com padronização
        st.subheader("Modelo com Padronização")
        col1, col2 = st.columns(2)
        
        with col1:
            y_pred_test = clf_def.predict(comp2_test)
            y_pred_train = clf_def.predict(comp2_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            
            acuracia_dos_modelos['acuracia_teste']['com_padronizacao'].append(acc_test)
            acuracia_dos_modelos['acuracia_treino']['com_padronizacao'].append(acc_train)
            
            st.metric("Acurácia Treino", f"{acc_train:.3f}")
            st.metric("Acurácia Teste", f"{acc_test:.3f}")
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_estimator(
                clf_def,
                comp2_test,
                y_test,
                display_labels=labels.label.tolist(),
                cmap=plt.cm.Blues,
                normalize="true",
                ax=ax
            )
            plt.title("Matriz de Confusão - Com Padronização")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

        # Resumo comparativo
        st.subheader("Resumo Comparativo")
        df_comparativo = pd.DataFrame(acuracia_dos_modelos).applymap(
            lambda x: [round(v, 3) for v in x] if isinstance(x, list) else round(x, 3)
        )
        st.dataframe(df_comparativo)