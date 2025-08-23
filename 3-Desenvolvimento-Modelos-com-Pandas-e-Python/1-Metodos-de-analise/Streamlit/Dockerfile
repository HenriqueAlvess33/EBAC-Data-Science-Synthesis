# Use Python 3.10
FROM python:3.10-slim

# Define diretório de trabalho
WORKDIR /app

# Copia os arquivos de dependências
COPY requirements.txt .

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código para o container
COPY . .

# Comando para rodar o Streamlit
CMD ["streamlit", "run", "script.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
