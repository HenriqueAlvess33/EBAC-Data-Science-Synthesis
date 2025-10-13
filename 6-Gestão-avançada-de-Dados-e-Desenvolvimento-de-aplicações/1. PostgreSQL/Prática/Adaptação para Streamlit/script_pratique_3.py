import streamlit as st
import psycopg2 as pg2
import pandas as pd

# Configuração da página
st.set_page_config(page_title="DVD Rental Analysis", layout="wide")

# Título da aplicação
st.title("📊 Análise DVD Rental")
st.markdown("---")

# Função para conectar ao banco de dados PostgreSQL
def connect_to_postgres(db):
    try:
        # Usando secrets do Streamlit para informações sensíveis
        conn = pg2.connect(
            host='localhost', 
            database=db, 
            user='postgres', 
            password='senha123'
        )
        return conn
    except Exception as e:
        st.error(f"Erro ao tentar se conectar ao PostgreSQL: {e}")
        return None

# Função para executar queries
def execute_query(conn, query):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        data = cursor.fetchall()
        colnames = [desc.name for desc in cursor.description]
        cursor.close()
        return data, colnames
    except Exception as e:
        st.error(f"Erro na execução da query: {e}")
        return None, None

# Função para converter query em DataFrame
def query_to_dataframe(conn, query):
    data, colnames = execute_query(conn, query)
    if data is not None:
        df = pd.DataFrame(data, columns=colnames)
        return df
    else:
        return None

# Sidebar para navegação
st.sidebar.title("Navegação")
analysis_type = st.sidebar.selectbox(
    "Selecione a Análise:",
    ["Análise de Atores", "Análise de Clientes", "Consulta Personalizada"]
)

# Conexão com o banco de dados
if st.sidebar.button("Conectar ao Banco de Dados"):
    with st.spinner("Conectando ao banco de dados..."):
        conn = connect_to_postgres('dvdrental')
        if conn:
            st.sidebar.success("Conectado com sucesso!")
            st.session_state.conn = conn
        else:
            st.sidebar.error("Falha na conexão")

# Análise de Atores
if analysis_type == "Análise de Atores":
    st.header("🎬 Análise de Atores e Filmes")
    
    if st.button("Carregar Dados de Atores"):
        if 'conn' not in st.session_state:
            st.warning("Por favor, conecte ao banco de dados primeiro.")
        else:
            with st.spinner("Carregando dados..."):
                query = '''
                SELECT first_name, last_name, 
                       AVG(rental_duration) as rental_duration, 
                       AVG(rental_rate) as rental_rate, 
                       AVG(length) as length, 
                       AVG(replacement_cost) as replacement_cost
                FROM (
                    SELECT * 
                    FROM film as x
                    LEFT JOIN film_actor as y
                        ON x.film_id = y.film_id
                    LEFT JOIN actor as z
                        ON y.actor_id = z.actor_id
                ) as a
                GROUP BY first_name, last_name
                ORDER BY rental_duration DESC
                '''
                
                df = query_to_dataframe(st.session_state.conn, query)
                
                if df is not None:
                    st.success("Dados carregados com sucesso!")
                    
                    # Exibir DataFrame
                    st.subheader("📋 Tabela de Dados")
                    st.dataframe(df, use_container_width=True)
                    
                    # Métricas principais
                    st.subheader("📈 Métricas Principais")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Maior Duração de Aluguel", 
                            f"{df['rental_duration'].max():.1f} dias"
                        )
                    
                    with col2:
                        st.metric(
                            "Maior Taxa de Aluguel", 
                            f"${df['rental_rate'].max():.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Filme Mais Longo", 
                            f"{df['length'].max():.0f} min"
                        )
                    
                    with col4:
                        st.metric(
                            "Maior Custo de Reposição", 
                            f"${df['replacement_cost'].max():.2f}"
                        )
                    
                    # Filtros interativos
                    st.subheader("🔍 Filtros")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        min_duration = st.slider(
                            "Duração Mínima de Aluguel (dias)",
                            min_value=float(df['rental_duration'].min()),
                            max_value=float(df['rental_duration'].max()),
                            value=float(df['rental_duration'].min())
                        )
                    
                    with col2:
                        min_rate = st.slider(
                            "Taxa Mínima de Aluguel ($)",
                            min_value=float(df['rental_rate'].min()),
                            max_value=float(df['rental_rate'].max()),
                            value=float(df['rental_rate'].min())
                        )
                    
                    # Aplicar filtros
                    filtered_df = df[
                        (df['rental_duration'] >= min_duration) & 
                        (df['rental_rate'] >= min_rate)
                    ]
                    
                    st.write(f"**Resultados filtrados:** {len(filtered_df)} atores")
                    st.dataframe(filtered_df, use_container_width=True)

# Análise de Clientes
elif analysis_type == "Análise de Clientes":
    st.header("👥 Análise de Clientes")
    
    if st.button("Carregar Dados de Clientes"):
        if 'conn' not in st.session_state:
            st.warning("Por favor, conecte ao banco de dados primeiro.")
        else:
            with st.spinner("Carregando dados de clientes..."):
                query = '''
                SELECT first_name, last_name, email, COUNT(rental_id) as total_alugueis
                FROM customer as x
                LEFT JOIN rental as y
                    ON x.customer_id = y.customer_id
                GROUP BY first_name, last_name, email
                ORDER BY total_alugueis DESC
                '''
                
                df = query_to_dataframe(st.session_state.conn, query)
                
                if df is not None:
                    st.success("Dados de clientes carregados com sucesso!")
                    
                    # Exibir DataFrame
                    st.subheader("📋 Clientes e Total de Aluguéis")
                    st.dataframe(df, use_container_width=True)
                    
                    # Métricas de clientes
                    st.subheader("📊 Estatísticas de Clientes")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total de Clientes", 
                            len(df)
                        )
                    
                    with col2:
                        st.metric(
                            "Maior Número de Aluguéis", 
                            int(df['total_alugueis'].max())
                        )
                    
                    with col3:
                        st.metric(
                            "Média de Aluguéis por Cliente", 
                            f"{df['total_alugueis'].mean():.1f}"
                        )
                    
                    # Top clientes
                    st.subheader("🏆 Top 10 Clientes")
                    top_10 = df.head(10)
                    st.dataframe(top_10, use_container_width=True)

# Consulta Personalizada
else:
    st.header("🔧 Consulta Personalizada")
    
    query_input = st.text_area(
        "Digite sua consulta SQL:",
        height=150,
        placeholder="SELECT * FROM table_name LIMIT 10;"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("Executar Consulta"):
            if query_input and 'conn' in st.session_state:
                with st.spinner("Executando consulta..."):
                    df = query_to_dataframe(st.session_state.conn, query_input)
                    if df is not None:
                        st.success("Consulta executada com sucesso!")
                        st.session_state.custom_df = df
    
    if 'custom_df' in st.session_state:
        df = st.session_state.custom_df
        
        st.subheader("📋 Resultados da Consulta")
        st.dataframe(df, use_container_width=True)
        
        st.subheader("📈 Estatísticas do Dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Forma do Dataset:**")
            st.write(f"Linhas: {df.shape[0]}")
            st.write(f"Colunas: {df.shape[1]}")
        
        with col2:
            st.write("**Tipos de Dados:**")
            st.write(df.dtypes.astype(str))

# Informações na sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Instruções:**
    1. Clique em 'Conectar ao Banco de Dados'
    2. Selecione o tipo de análise
    3. Clique no botão para carregar os dados
    """
)