# model_final.py
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

class CreditScoringModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.features = None
        
    def load_model(self, model_path='model_final.pkl'):
        """Carrega o modelo treinado"""
        try:
            with open(model_path, 'rb') as file:
                loaded_data = pickle.load(file)
                
            # Verificar se é um dicionário com modelo e features
            if isinstance(loaded_data, dict):
                self.model = loaded_data.get('model')
                self.features = loaded_data.get('features', [])
                self.preprocessor = loaded_data.get('preprocessor')
            else:
                self.model = loaded_data
                
            return True if self.model is not None else False
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return False
    
    def preprocess_data(self, df):
        """Pré-processa os dados para escoragem"""
        # Fazer uma cópia do dataframe
        df_processed = df.copy()
        
        # Tratamento de missings
        if 'tempo_emprego' in df_processed.columns:
            df_processed['tempo_emprego'] = df_processed['tempo_emprego'].fillna(-1)
        
        # Engenharia de features
        if 'data_ref' in df_processed.columns:
            df_processed['data_ref'] = pd.to_datetime(df_processed['data_ref'])
            df_processed['mes'] = df_processed['data_ref'].dt.month
            df_processed['ano'] = df_processed['data_ref'].dt.year
        
        # Criar variáveis dummy para categorias importantes
        categorical_vars = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 
                           'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia']
        
        for var in categorical_vars:
            if var in df_processed.columns:
                # Criar dummies mantendo todas as categorias
                dummies = pd.get_dummies(df_processed[var], prefix=var)
                df_processed = pd.concat([df_processed, dummies], axis=1)
        
        # Remover colunas originais categóricas
        df_processed = df_processed.drop(columns=categorical_vars, errors='ignore')
        
        # Garantir que todas as features esperadas pelo modelo estejam presentes
        expected_features = self.get_expected_features()
        for feature in expected_features:
            if feature not in df_processed.columns:
                df_processed[feature] = 0
        
        # Manter apenas as features esperadas pelo modelo
        df_processed = df_processed[expected_features]
        
        return df_processed
    
    def get_expected_features(self):
        """Retorna a lista de features esperadas pelo modelo"""
        # Features base que devem estar presentes
        base_features = [
            'idade', 'tempo_emprego', 'renda', 'qtd_filhos', 'qt_pessoas_residencia',
            'mes', 'ano'
        ]
        
        # Features categóricas dummy
        categorical_dummies = [
            'sexo_F', 'sexo_M',
            'posse_de_veiculo_N', 'posse_de_veiculo_Y',
            'posse_de_imovel_N', 'posse_de_imovel_Y',
            'tipo_renda_Assalariado', 'tipo_renda_Empresário', 'tipo_renda_Pensionista', 
            'tipo_renda_Servidor público',
            'educacao_Pós graduação', 'educacao_Superior completo', 'educacao_Superior incompleto',
            'educacao_Médio',
            'estado_civil_Casado', 'estado_civil_Separado', 'estado_civil_Solteiro',
            'estado_civil_União', 'estado_civil_Viúvo',
            'tipo_residencia_Aluguel', 'tipo_residencia_Casa', 'tipo_residencia_Com os pais',
            'tipo_residencia_Governamental'
        ]
        
        return base_features + [f for f in categorical_dummies if f in [
            'sexo_M', 'posse_de_veiculo_Y', 'posse_de_imovel_Y',
            'tipo_renda_Empresário', 'tipo_renda_Pensionista', 'tipo_renda_Servidor público',
            'educacao_Superior completo', 'estado_civil_Separado', 'estado_civil_Solteiro',
            'estado_civil_União', 'estado_civil_Viúvo', 'tipo_residencia_Casa', 
            'tipo_residencia_Com os pais', 'tipo_residencia_Governamental'
        ]]
    
    def predict(self, df):
        """Faz previsões na base de dados"""
        if self.model is None:
            raise ValueError("Modelo não carregado. Chame load_model() primeiro.")
        
        # Pré-processar dados
        df_processed = self.preprocess_data(df)
        
        try:
            # Verificar métodos disponíveis no modelo
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(df_processed)[:, 1]
            else:
                # Fallback para modelos sem predict_proba
                probabilities = self.model.predict(df_processed)
            
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(df_processed)
            else:
                predictions = (probabilities > 0.068).astype(int)
            
            # Adicionar resultados ao dataframe original
            result_df = df.copy()
            result_df['score'] = probabilities
            result_df['prediction'] = predictions
            result_df['classificacao'] = result_df['score'].apply(
                lambda x: 'Alto Risco' if x > 0.068 else 'Baixo Risco'
            )
            
            return result_df
            
        except Exception as e:
            raise Exception(f"Erro durante predição: {str(e)}")
    
    def get_model_info(self):
        """Retorna informações sobre o modelo"""
        if self.model is None:
            return {
                'model_type': 'Não carregado',
                'features': [],
                'n_features': 0
            }
        
        info = {
            'model_type': type(self.model).__name__,
            'features': self.get_expected_features(),
            'n_features': len(self.get_expected_features())
        }
        return info

# Função para criar modelo de exemplo se o arquivo não existir
def create_sample_model():
    """Cria um modelo de exemplo para demonstração"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Gerar dados de exemplo
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    
    # Treinar modelo simples
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Criar estrutura com modelo e features
    model_data = {
        'model': model,
        'features': [f'feature_{i}' for i in range(10)],
        'preprocessor': None
    }
    
    # Salvar modelo
    with open('model_final.pkl', 'wb') as file:
        pickle.dump(model_data, file)
    
    print("✅ Modelo de exemplo criado com sucesso!")
    return True

def create_sample_data():
    """Cria dados de exemplo para teste"""
    np.random.seed(42)
    
    sample_data = {
        'data_ref': ['2015-01-01'] * 10,
        'sexo': np.random.choice(['M', 'F'], 10),
        'posse_de_veiculo': np.random.choice(['Y', 'N'], 10),
        'posse_de_imovel': np.random.choice(['Y', 'N'], 10),
        'qtd_filhos': np.random.randint(0, 4, 10),
        'tipo_renda': np.random.choice(['Assalariado', 'Empresário', 'Servidor público', 'Pensionista'], 10),
        'educacao': np.random.choice(['Superior completo', 'Superior incompleto', 'Médio'], 10),
        'estado_civil': np.random.choice(['Casado', 'Solteiro', 'União', 'Separado', 'Viúvo'], 10),
        'tipo_residencia': np.random.choice(['Casa', 'Apartamento', 'Com os pais', 'Governamental'], 10),
        'idade': np.random.randint(25, 65, 10),
        'tempo_emprego': np.random.uniform(1, 20, 10),
        'qt_pessoas_residencia': np.random.randint(1, 6, 10),
        'renda': np.random.uniform(2000, 10000, 10)
    }
    
    return pd.DataFrame(sample_data)
