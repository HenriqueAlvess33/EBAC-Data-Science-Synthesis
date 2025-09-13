FROM python:3.10-slim

# Diretório de trabalho
WORKDIR /app

# Instalar ferramentas de build
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Rodar Streamlit na porta dinâmica
CMD ["streamlit", "run", "script.py", "--server.port", "$PORT", "--server.address", "0.0.0.0"]

