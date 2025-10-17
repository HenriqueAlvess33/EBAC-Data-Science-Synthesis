# create_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


def create_and_save_model():
    """Cria e salva um modelo de exemplo"""
    # Criar dados de exemplo
    np.random.seed(42)
    n_samples = 1000

    X = pd.DataFrame(
        {
            "idade": np.random.randint(25, 65, n_samples),
            "tempo_emprego": np.random.uniform(1, 20, n_samples),
            "renda": np.random.uniform(2000, 10000, n_samples),
            "qtd_filhos": np.random.randint(0, 4, n_samples),
            "qt_pessoas_residencia": np.random.randint(1, 6, n_samples),
            "mes": np.random.randint(1, 13, n_samples),
            "ano": np.random.randint(2015, 2017, n_samples),
            "sexo_M": np.random.randint(0, 2, n_samples),
            "posse_de_veiculo_Y": np.random.randint(0, 2, n_samples),
            "posse_de_imovel_Y": np.random.randint(0, 2, n_samples),
            "tipo_renda_Empresário": np.random.randint(0, 2, n_samples),
            "tipo_renda_Pensionista": np.random.randint(0, 2, n_samples),
            "tipo_renda_Servidor público": np.random.randint(0, 2, n_samples),
            "educacao_Superior completo": np.random.randint(0, 2, n_samples),
            "estado_civil_Separado": np.random.randint(0, 2, n_samples),
            "estado_civil_Solteiro": np.random.randint(0, 2, n_samples),
            "estado_civil_União": np.random.randint(0, 2, n_samples),
            "estado_civil_Viúvo": np.random.randint(0, 2, n_samples),
            "tipo_residencia_Casa": np.random.randint(0, 2, n_samples),
            "tipo_residencia_Com os pais": np.random.randint(0, 2, n_samples),
            "tipo_residencia_Governamental": np.random.randint(0, 2, n_samples),
        }
    )

    # Criar target (mau) baseado em algumas regras
    y = ((X["renda"] < 3000) | (X["tempo_emprego"] < 2) | (X["idade"] < 25)).astype(int)

    # Treinar modelo
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
    )

    model.fit(X, y)

    # Salvar modelo com estrutura esperada
    model_data = {"model": model, "features": X.columns.tolist(), "preprocessor": None}

    with open("model_final.pkl", "wb") as file:
        pickle.dump(model_data, file)

    print("✅ Modelo criado e salvo com sucesso!")
    print(f"📊 Acurácia: {model.score(X, y):.3f}")
    print(f"🔢 Número de features: {len(X.columns)}")


if __name__ == "__main__":
    create_and_save_model()
