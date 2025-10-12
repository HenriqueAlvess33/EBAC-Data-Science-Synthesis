import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuração da página
st.set_page_config(page_title="Análise de Dados de Sensor", layout="wide")

# Título da aplicação
st.title("📊 Análise de Dados de Sensor")
st.markdown("---")

# Área para upload de arquivos
st.header("📁 Upload de Dados")
uploaded_file = st.file_uploader(
    "Faça upload do arquivo CSV com dados do sensor", 
    type=['csv'],
    help="Arquivo deve conter colunas: time, power, temp, humidity, light, CO2, dust"
)

if uploaded_file is not None:
    try:
        # Carregar dados
        sensor_data = pd.read_csv(uploaded_file)
        
        # Sidebar para configurações
        st.sidebar.header("Configurações de Análise")
        
        # Mostrar informações básicas do dataset
        st.subheader("📋 Visualização dos Dados")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Primeiras linhas do dataset:**")
            st.dataframe(sensor_data.head())
        
        with col2:
            st.write("**Informações do dataset:**")
            st.write(f"Shape: {sensor_data.shape}")
            st.write(f"Colunas: {list(sensor_data.columns)}")
        
        # Pré-processamento dos dados
        st.subheader("🔧 Pré-processamento dos Dados")
        
        # Verificar duplicatas no tempo
        duplicates = sensor_data[sensor_data.time.duplicated(keep=False)]
        if not duplicates.empty:
            st.warning(f"Encontradas {len(duplicates)} linhas com tempo duplicado")
            st.dataframe(duplicates.head())
            
            # Opção para remover duplicatas
            if st.checkbox("Remover linhas com tempo duplicado", value=True):
                sensor_data = sensor_data.drop_duplicates(subset='time', keep='first')
                st.success("Duplicatas removidas!")
        
        # Converter coluna time para datetime e definir como índice
        sensor_data = (sensor_data
                      .assign(time=pd.to_datetime(sensor_data['time']))
                      .set_index('time'))
        
        # Visualização dos dados originais
        st.subheader("📈 Visualização dos Dados dos Sensores")
        
        # Seleção de sensores para visualizar
        available_sensors = ['power', 'temp', 'humidity', 'light', 'CO2', 'dust']
        selected_sensors = st.multiselect(
            "Selecione os sensores para visualizar:",
            available_sensors,
            default=available_sensors
        )
        
        if selected_sensors:
            fig, axes = plt.subplots(len(selected_sensors), 1, figsize=(15, 4*len(selected_sensors)))
            if len(selected_sensors) == 1:
                axes = [axes]
            
            for i, sensor in enumerate(selected_sensors):
                axes[i].plot(sensor_data.index, sensor_data[sensor])
                axes[i].set_title(f'Sensor: {sensor}')
                axes[i].set_ylabel(sensor)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Análise de Médias Móveis
        st.subheader("📊 Análise de Médias Móveis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sensor_for_ma = st.selectbox(
                "Selecione o sensor para análise de média móvel:",
                available_sensors
            )
        
        with col2:
            window_size = st.slider(
                "Tamanho da janela para média móvel:",
                min_value=10,
                max_value=1000,
                value=250,
                step=10
            )
        
        if sensor_for_ma:
            fig, ax = plt.subplots(figsize=(15, 6))
            sns.set()
            
            # Plot dos dados originais e média móvel
            sensor_data[sensor_for_ma].plot(ax=ax, label='Original', alpha=0.7)
            sensor_data[sensor_for_ma].rolling(window_size).mean().plot(
                ax=ax, 
                label=f'Média Móvel ({window_size} pontos)',
                linewidth=2
            )
            
            ax.set_title(f'Dados do Sensor {sensor_for_ma} - Original vs Média Móvel')
            ax.set_ylabel(sensor_for_ma)
            ax.legend()
            ax.grid(True, alpha=0.3)
            sns.despine()
            
            st.pyplot(fig)
        
        # Resample dos dados
        st.subheader("⏱️ Reamostragem dos Dados")
        
        resample_freq = st.selectbox(
            "Frequência para reamostragem:",
            ['1min', '5min', '10min', '30min', '1H', '2H', '6H', '12H', '1D'],
            index=4
        )
        
        method = st.radio(
            "Método de reamostragem:",
            ['mean', 'ffill', 'bfill']
        )
        
        if st.button("Aplicar Reamostragem"):
            if method == 'mean':
                resampled_data = sensor_data.resample(resample_freq).mean()
            elif method == 'ffill':
                resampled_data = sensor_data.resample(resample_freq).ffill()
            else:
                resampled_data = sensor_data.resample(resample_freq).bfill()
            
            st.write(f"Dados reamostrados ({resample_freq} - {method}):")
            st.dataframe(resampled_data.head(10))
            
            # Plot dos dados reamostrados
            if len(selected_sensors) > 0:
                fig, axes = plt.subplots(len(selected_sensors), 1, figsize=(15, 4*len(selected_sensors)))
                if len(selected_sensors) == 1:
                    axes = [axes]
                
                for i, sensor in enumerate(selected_sensors):
                    axes[i].plot(resampled_data.index, resampled_data[sensor])
                    axes[i].set_title(f'Sensor {sensor} - Reamostrado ({resample_freq})')
                    axes[i].set_ylabel(sensor)
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Análise de Lag
        st.subheader("🔄 Análise de Lag (Deslocamento Temporal)")
        
        lag_value = st.slider(
            "Valor do lag (deslocamento):",
            min_value=1,
            max_value=10,
            value=1
        )
        
        sensor_for_lag = st.selectbox(
            "Sensor para análise de lag:",
            available_sensors
        )
        
        if sensor_for_lag:
            lag_col_name = f'{sensor_for_lag}_lag_{lag_value}'
            sensor_data[lag_col_name] = sensor_data[sensor_for_lag].shift(+lag_value)
            
            st.write(f"Comparação entre {sensor_for_lag} original e com lag {lag_value}:")
            comparison_cols = [sensor_for_lag, lag_col_name]
            st.dataframe(sensor_data[comparison_cols].head(15))
            
            # Plot comparativo
            fig, ax = plt.subplots(figsize=(15, 6))
            sensor_data[sensor_for_lag].plot(ax=ax, label=f'{sensor_for_lag} Original', alpha=0.7)
            sensor_data[lag_col_name].plot(ax=ax, label=f'{sensor_for_lag} Lag {lag_value}', alpha=0.7)
            ax.set_title(f'Comparação: {sensor_for_lag} Original vs Lag {lag_value}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Estatísticas descritivas
        st.subheader("📊 Estatísticas Descritivas")
        st.dataframe(sensor_data[available_sensors].describe())
        
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {str(e)}")
        st.info("Verifique se o arquivo CSV possui o formato correto e a coluna 'time'.")

else:
    st.info("👆 Por favor, faça upload de um arquivo CSV para começar a análise.")
    
    # Exemplo de estrutura esperada
    st.subheader("Estrutura esperada do arquivo CSV:")
    example_data = {
        'time': ['2015-08-16 05:08:23', '2015-08-16 05:08:24', '2015-08-16 05:08:25'],
        'power': [120.5, 121.0, 119.8],
        'temp': [25.3, 25.4, 25.2],
        'humidity': [45.2, 45.1, 45.3],
        'light': [320, 325, 315],
        'CO2': [450, 455, 448],
        'dust': [27.8, 27.1, 28.2]
    }
    st.dataframe(pd.DataFrame(example_data))