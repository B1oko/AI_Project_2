import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import pickle
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim

from utils.Preprocessor import Preprocessor
from models.models import RegressionNN
from utils.data_options import OPTIONS_GENDER, OPTIONS_EDUCATION_LEVEL, OPTIONS_JOB_CATEGORY, MAP_JOB_TYPE_JOB_TITLE


# Load the preprocessor
preprocessor = Preprocessor()

def app():
    st.title('Predicción de Salarios')

    # Obtener los datos de entrada del usuario
    st.write("Completa el siguiente formulario para obtener una estimación de tu salario:")

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
    st.write("Elige el modelo con el que quieres obtener la predicción:")

    col1, col2 = st.columns(2)
    with col1:
        rf_button = st.button("Random Forest")
    with col2:
        nn_button = st.button("Neural Network")

    @st.cache_resource
    def cargar_modelo(rf=True):
        if rf:
            with open("models/regression_rf.pkl", "rb") as file:
                return pickle.load(file)
        else:
            model = RegressionNN()
            model.load_state_dict(torch.load('models/regression_nn.pth', weights_only=True))
            model.eval()
            return model


    # Ejecutar el modelo según el botón pulsado
    if rf_button:

        if not job_type:
            st.write("Por favor seleccione un tipo de trabajo.")
            return

        # Preprocessing
        data_iference = preprocessor.transform([[age+i, gender, education_level, job_title, years_of_experience+i, job_category, job_type] for i in range(20)])
        
        # Cargar modelo Random Forest (ejemplo con un modelo preentrenado)
        rf_model = cargar_modelo()
        
        y_pred_rf = rf_model.predict(data_iference)
        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <span style="font-size:16px; color:#ffffff; margin-right:10px;">Predicción del modelo con Random Forest:</span>
                <span style="
                    font-size:20px; 
                    color:#00ffa6; 
                    background-color:#292e36; 
                    padding:4px 8px; 
                    border-radius:5px;
                ">
                    {y_pred_rf[0]:.2f} $
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # plot y_pred
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        ax.plot(y_pred_rf, color='#00ffa6',label='Inferencia')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Salario')
        ax.set_title('Salario en los proximos años.')
        # Cambiar el color de los textos a blanco
        ax.tick_params(colors='white')  # Color de los ticks (números de los ejes)
        ax.xaxis.label.set_color('white')  # Color del texto del eje X
        ax.yaxis.label.set_color('white')  # Color del texto del eje Y
        ax.title.set_color('white')  # Color del título
        st.pyplot(fig, transparent=True)
    

    if nn_button:

        if not job_type:
            st.write("Por favor seleccione un tipo de trabajo.")
            return

        # Preprocessing
        data_iference = preprocessor.transform([[age+i, gender, education_level, job_title, years_of_experience+i, job_category, job_type] for i in range(20)])
        
        model = cargar_modelo(rf=False)
        
        # Realizar la predicción del salario
        with torch.no_grad():
            y_pred_nn = model(torch.tensor(data_iference, dtype=torch.float32))

        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <span style="font-size:16px; color:#ffffff; margin-right:10px;">Predicción del modelo con Neural Network:</span>
                <span style="
                    font-size:20px; 
                    color:#00ffa6; 
                    background-color:#292e36; 
                    padding:4px 8px; 
                    border-radius:5px;
                ">
                    {y_pred_nn[0, 0].item():.2f} $
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        ax.plot(y_pred_nn, color='#00ffa6',label='Inferencia')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Salario')
        ax.set_title('Salario en los proximos años.')
        # Cambiar el color de los textos a blanco
        ax.tick_params(colors='white')  # Color de los ticks (números de los ejes)
        ax.xaxis.label.set_color('white')  # Color del texto del eje X
        ax.yaxis.label.set_color('white')  # Color del texto del eje Y
        ax.title.set_color('white')  # Color del título
        st.pyplot(fig, transparent=True)
