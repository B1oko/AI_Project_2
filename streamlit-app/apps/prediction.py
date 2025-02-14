import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from models.models import RegressionNN

from utils import Preprocessor
from utils.data_options import OPTIONS_GENDER, OPTIONS_EDUCATION_LEVEL, OPTIONS_JOB_CATEGORY, MAP_JOB_TYPE_JOB_TITLE


'''@st.cache_resource
def load_model():
    model = RegressionNN()
    model.load_state_dict(torch.load('models/regression_nn.pth', weights_only=True))
    model.eval()
    return model

model = load_model()
'''

with open("models/preprocessing_pipeline.pkl", "rb") as f:
    preprocessor = pickle.load(f)

def app():
    st.title('Predicciones')

    st.write("Aqui se permitira seleccionar variables para realizar una predicción")

    # Obtener los datos de entrada del usuario
    st.write("Ingrese los datos de entrada:")

    age =  st.slider('Edad', 18, 100, 18)
    gender = st.selectbox("Sexo", OPTIONS_GENDER)
    education_level = st.selectbox("Nivel de educación", OPTIONS_EDUCATION_LEVEL)
    years_of_experience = st.slider("Años de experiencia", 0, 50, 0)

    job_category = st.selectbox("Categoría del trabajo", OPTIONS_JOB_CATEGORY)
    job_type = st.selectbox("Tipo de trabajo:", list(MAP_JOB_TYPE_JOB_TITLE.keys()), None)
    if job_type:
        job_title = st.selectbox("Nombre del trabajo", MAP_JOB_TYPE_JOB_TITLE[job_type])

    # Añadimos un espacio
    st.write("")
    
    # Botones para elegir el modelo
    st.write("Elige el modelo que quieres ejecutar:")

    col1, col2 = st.columns(2)
    with col1:
        rf_button = st.button("Random Forest")
    with col2:
        nn_button = st.button("Neural Network")

    # Ejecutar el modelo según el botón pulsado
    if rf_button:

        if not job_type:
            st.write("Por favor seleccione un tipo de trabajo.")
            return

        # Preprocessing
        X_inference = [age, gender, education_level, job_title, years_of_experience, job_category, job_type]
        df_inference = pd.DataFrame([X_inference], columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Job Category", "Job Type"])
        X_inference_transformed = preprocessor.transform(df_inference)

        st.write("Ejecutando modelo de Random Forest...")
        
        # Cargar modelo Random Forest (ejemplo con un modelo preentrenado)
        with open("models/regression_rf.pkl", "rb") as file:
            rf_model = pickle.load(file)
        
        y_pred_rf = rf_model.predict(X_inference_transformed)
        st.write("Predicciones del modelo Random Forest:", y_pred_rf)
    

    if nn_button:

        if not job_type:
            st.write("Por favor seleccione un tipo de trabajo.")
            return

        # Preprocessing
        X_inference = [age, gender, education_level, job_title, years_of_experience, job_category, job_type]
        df_inference = pd.DataFrame([X_inference], columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Job Category", "Job Type"])
        X_inference_transformed = preprocessor.transform(df_inference)

        st.write("Ejecutando modelo de Neural Network...")
        
        # Cargar modelo Neural Network (ejemplo con un modelo preentrenado)
        model = RegressionNN()
        model.load_state_dict(torch.load('models/regression_nn.pth', weights_only=True))
        model.eval()
        
        # Realizar la predicción del salario
        with torch.no_grad():
            y_pred_nn = model(torch.tensor(X_inference_transformed, dtype=torch.float32).unsqueeze(0))

        st.write("Predicciones del modelo Neural Network:", y_pred_nn)
