├── script_pratique_2.py                  # Aplicação principal Streamlit
├── model_final.py         			# Classe do modelo e funções auxiliares
├── create_model.py       		 	# Script para criar modelo de exemplo
├── model_final.pkl        			# Modelo treinado (gerado automaticamente)

Crie o modelo inicial (se necessário)
No terminal: `python create_model.py`

Acesse no navegador
http://localhost:8501

 Funcionalidades

Análise de Dados

    Upload de CSV: Carregamento de arquivos de dados dos clientes

    Visualização Interativa: Tabelas e estatísticas descritivas

    Validação de Dados: Verificação automática de colunas necessárias

Escoragem Automática

    Processamento em Lote: Escoragem de múltiplos clientes simultaneamente

    Modelo de ML: Random Forest para classificação de risco

    Threshold Configurável: Ponto de corte em 0.068 para classificação

Visualização de Resultados

    Distribuição de Scores: Histograma interativo dos resultados

    Análise de Risco: Proporção entre Alto/Baixo Risco

    Comparação por Variáveis: Boxplots por características demográficas

Exportação

    Formatos Múltiplos: CSV e Excel

    Relatório Executivo: Resumo estatístico dos resultados

    Timestamp Automático: Nomes de arquivo com data/hora