import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Definir un pipeline de preprocesamiento
numeric_features = [0, 1, 2]  # Columnas numéricas
categorical_features = [3, 4]  # Columnas categóricas

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features),
    ]
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
X_train = np.random.rand(100, 5)
pipeline.fit(X_train)

# Guardar el pipeline
with open("preprocessing_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
