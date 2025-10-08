import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st



# Criar um script que seja capaz de receber quantidade irrestrita de argumentos e processar diferentes dataframes
def plota_pivot_table(dataframe, values, index, funcao, ylabel, xlabel, opcao="nada"):

    if opcao == "nada":
        pd.pivot_table(dataframe, values=values, index=index, aggfunc=funcao).plot(
            figsize=[15, 5]
        )
    elif opcao == "unstack":
        pd.pivot_table(
            dataframe, values=values, index=index, aggfunc=funcao
        ).unstack().plot(figsize=[18, 6])
    elif opcao == "sort_values":
        pd.pivot_table(
            dataframe, values=values, index=index, aggfunc=funcao
        ).sort_values(values).plot(figsize=[15, 5])

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    st.pyplot(fig=plt)

    return None

# Configuração para o conteúdo se distribuir por toda a extensão da página e não ficar totalmente centralizado com as laterais livres
st.set_page_config(page_title= 'SINASC Rondônia', layout='wide', page_icon= 'https://seeklogo.com/images/B/bandeira-rondonia-logo-FEC40A3CCA-seeklogo.com.png')

st.write('# Análise SINASC')

sinasc = pd.read_csv('./input/SINASC_RO_2019.csv')

sinasc.DTNASC = pd.to_datetime(sinasc.DTNASC)

# Definição de valores mínimos e máximos para escrita na interface
min_data = sinasc.DTNASC.min()
max_data = sinasc.DTNASC.max()

# Escrevendo na tela a menor data e maior data, demonstrando o alcance do dataset
st.write(min_data)
st.write(max_data)

# Ordenação de valores da coluna DTNASC em ordem e sem repetições, reprodução da coluna na interface
datas = pd.DataFrame(sinasc.DTNASC.unique(), columns=['DTNASC'])
datas.sort_values(by='DTNASC', inplace=True, ignore_index=True)
st.write(datas)

# Construção de sistema que possibilita a filtragem por parte do usuário 
data_incial = st.sidebar.date_input('Data inicial', value=min_data, min_value=min_data, max_value=max_data)
data_final = st.sidebar.date_input('Data final', value=max_data, min_value=min_data, max_value=max_data)

# Confirmação escrita na tela da data selecionada
st.write('Data inicial =', data_incial)
st.write('Data final =', data_final)

# Criação de um dataframe que sofra influencia da filtragem realizada pelo usuário
teste = sinasc[(sinasc['DTNASC']<= pd.to_datetime(data_final)) & (sinasc['DTNASC'] >= pd.to_datetime(data_incial))]
st.write(teste)
st.write(teste.shape)

# Alteração do nome de teste para sinasc para as alterações de filtragem serem aplicadas diretamente no dataframe e influenciarem as visualizações de imediato, caso contrário as plotagens não sofreriam as filtragens
sinasc = sinasc[(sinasc['DTNASC']<= pd.to_datetime(data_final)) & (sinasc['DTNASC'] >= pd.to_datetime(data_incial))]
st.write(sinasc)
st.write(sinasc.shape)

dicionario_de_parametros = [
    {
        "values": "IDADEMAE",
        "index": "DTNASC",
        "funcao": "count",
        "ylabel": "Quantidade de nascimentos",
        "xlabel": "Datas de nascimentos",
        "opcao": "nada",
        "nome_do_arquivo": "media_idade_mae.png",
    },
    {
        "values": "IDADEMAE",
        "index": "DTNASC",
        "funcao": "mean",
        "ylabel": "Idade das mães",
        "xlabel": "Datas de nascimentos",
        "opcao": "nada",
        "nome_do_arquivo": "contagem_nascimentos_por_data.png",
    },
    {
        "values": "IDADEMAE",
        "index": ["DTNASC", "SEXO"],
        "funcao": "count",
        "ylabel": "Idade das mães",
        "xlabel": "Datas de nascimentos",
        "opcao": "unstack",
        "nome_do_arquivo": "contagem_nascimentos_por_data_e_sexo.png",
    },
    {
        "values": "PESO",
        "index": ["DTNASC", "SEXO"],
        "funcao": "count",
        "ylabel": "Idade das mães",
        "xlabel": "Datas de nascimentos",
        "opcao": "unstack",
        "nome_do_arquivo": "peso_recem_nascidos_por_data_e_sexo.png",
    },
    {
        "values": "PESO",
        "index": "ESCMAE",
        "funcao": "median",
        "ylabel": "Peso do bebê",
        "xlabel": "Tempo de escolaridade das mães",
        "opcao": "nada",
        "nome_do_arquivo": "peso_por_escolaridade.png",
    },
    {
        "values": "APGAR1",
        "index": "GESTACAO",
        "funcao": "mean",
        "ylabel": "Valor de Apgar 1",
        "xlabel": "Semanas de gestação",
        "opcao": "sort_values",
        "nome_do_arquivo": "apgar1_por_gestacao.png",
    },
    {
        "values": "APGAR5",
        "index": "GESTACAO",
        "funcao": "mean",
        "ylabel": "Valor de Apgar 5",
        "xlabel": "Semanas de gestação",
        "opcao": "sort_values",
        "nome_do_arquivo": "apgar5_por_gestacao.png",
    },
]




for parametros in dicionario_de_parametros:
    plota_pivot_table(
        sinasc,
        values=parametros["values"],
        index=parametros["index"],
        funcao=parametros["funcao"],
        ylabel=parametros["ylabel"],
        xlabel=parametros["xlabel"],
        opcao=parametros["opcao"],
    )
