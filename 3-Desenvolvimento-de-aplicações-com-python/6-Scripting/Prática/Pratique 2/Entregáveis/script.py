import pandas as pd
import matplotlib.pyplot as plt
import os

import sys

# Criar um script que seja capaz de receber quantidade irrestrita de argumentos e processar diferentes dataframes

lista_de_argumentos = []

for argumento in sys.argv:
    lista_de_argumentos.append(argumento)

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

    return None

os.makedirs("./input/", exist_ok=True)
os.makedirs("./output/figs/", exist_ok=True)

input_dir = "./input/"
output_dir = "./output/figs/"

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



meses = sys.argv[1:]

for mes in meses:
    arquivo = f'SINASC_RO_2019_{mes}.csv'
    arquivo_path = os.path.join(input_dir, arquivo)
    sinasc = pd.read_csv(arquivo_path)
    max_data = sinasc.DTNASC.max()[:7]

    save_path = os.path.join(output_dir, max_data)
    os.makedirs(save_path, exist_ok=True)

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
        plt.savefig(os.path.join(save_path, parametros["nome_do_arquivo"]))
        plt.close()