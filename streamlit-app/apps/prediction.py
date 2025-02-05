import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim

from models.models import RegressionNN

GENDER_ENCODING = {'Female': 0, 'Male': 1, 'Other': 2}
EDUCATION_LEVEL_ENCODING = {
    "Bachelor's": 0,
    "Bachelor's Degree": 1,
    'High School': 2,
    "Master's": 3,
    "Master's Degree": 4,
    'PhD': 5
}
JOB_TITLE_ENCODING = {
    'Account Executive': 0,
    'Account Manager': 1,
    'Accountant': 2,
    'Administrative Assistant': 3,
    'Advertising Coordinator': 4,
    'Back end Developer': 5,
    'Business Analyst': 6,
    'Business Development Associate': 7,
    'Business Development Manager': 8,
    'Business Intelligence Analyst': 9,
    'Business Operations Analyst': 10,
    'CEO': 11,
    'Chief Data Officer': 12,
    'Chief Technology Officer': 13,
    'Consultant': 14,
    'Content Marketing Manager': 15,
    'Copywriter': 16,
    'Creative Director': 17,
    'Customer Service Manager': 18,
    'Customer Service Rep': 19,
    'Customer Service Representative': 20,
    'Customer Success Manager': 21,
    'Customer Success Rep': 22,
    'Customer Support Specialist': 23,
    'Data Analyst': 24,
    'Data Engineer': 25,
    'Data Entry Clerk': 26,
    'Data Scientist': 27,
    'Delivery Driver': 28,
    'Designer': 29,
    'Developer': 30,
    'Digital Content Producer': 31,
    'Digital Marketing Manager': 32,
    'Digital Marketing Specialist': 33,
    'Director': 34,
    'Director of Business Development': 35,
    'Director of Data Science': 36,
    'Director of Engineering': 37,
    'Director of Finance': 38,
    'Director of HR': 39,
    'Director of Human Capital': 40,
    'Director of Human Resources': 41,
    'Director of Marketing': 42,
    'Director of Operations': 43,
    'Director of Product Management': 44,
    'Director of Sales': 45,
    'Director of Sales and Marketing': 46,
    'Engineer': 47,
    'Event Coordinator': 48,
    'Financial Advisor': 49,
    'Financial Analyst': 50,
    'Financial Manager': 51,
    'Front End Developer': 52,
    'Front end Developer': 53,
    'Full Stack Engineer': 54,
    'Graphic Designer': 55,
    'HR Coordinator': 56,
    'HR Generalist': 57,
    'HR Manager': 58,
    'HR Specialist': 59,
    'Help Desk Analyst': 60,
    'Human Resources Coordinator': 61,
    'Human Resources Director': 62,
    'Human Resources Manager': 63,
    'Human Resources Specialist': 64,
    'IT Consultant': 65,
    'IT Manager': 66,
    'IT Project Manager': 67,
    'IT Support': 68,
    'IT Support Specialist': 69,
    'Juniour HR Coordinator': 70,
    'Juniour HR Generalist': 71,
    'Manager': 72,
    'Marketing Analyst': 73,
    'Marketing Coordinator': 74,
    'Marketing Director': 75,
    'Marketing Manager': 76,
    'Marketing Specialist': 77,
    'Network Engineer': 78,
    'Office Manager': 79,
    'Operations Analyst': 80,
    'Operations Coordinator': 81,
    'Operations Director': 82,
    'Operations Manager': 83,
    'Principal Engineer': 84,
    'Principal Scientist': 85,
    'Product Designer': 86,
    'Product Development Manager': 87,
    'Product Manager': 88,
    'Product Marketing Manager': 89,
    'Project Coordinator': 90,
    'Project Engineer': 91,
    'Project Manager': 92,
    'Public Relations Manager': 93,
    'Quality Assurance Analyst': 94,
    'Receptionist': 95,
    'Recruiter': 96,
    'Research Director': 97,
    'Research Scientist': 98,
    'Researcher': 99,
    'Sales Associate': 100,
    'Sales Director': 101,
    'Sales Executive': 102,
    'Sales Manager': 103,
    'Sales Operations Manager': 104,
    'Sales Representative': 105,
    'Scientist': 106,
    'Social Media Man': 107,
    'Social Media Manager': 108,
    'Social Media Specialist': 109,
    'Software Architect': 110,
    'Software Developer': 111,
    'Software Engineer': 112,
    'Software Engineer Manager': 113,
    'Software Manager': 114,
    'Software Project Manager': 115,
    'Strategy Consultant': 116,
    'Supply Chain Analyst': 117,
    'Supply Chain Manager': 118,
    'Technical Recruiter': 119,
    'Technical Support Specialist': 120,
    'Technical Writer': 121,
    'Training Specialist': 122,
    'UX Designer': 123,
    'UX Researcher': 124,
    'VP of Finance': 125,
    'VP of Operations': 126,
    'Web Designer': 127,
    'Web Developer': 128
}
SENIORITY_ENCODING = {
    'Analyst': 0,
    'Associate': 1,
    'Consultant': 2,
    'Developer': 3,
    'Director': 4,
    'Engineer': 5,
    'Junior': 6,
    'Manager': 7,
    'Regular': 8,
    'Scientist': 9,
    'Senior': 10,
    'Specialist': 11
}

model = RegressionNN()
model.load_state_dict(torch.load('models/regression_nn.pth', weights_only=True))
model.eval()


def app():
    st.title('Predicciones')

    st.write("Aqui se permitira seleccionar variables para realizar una predicción")

    # Obtener los datos de entrada del usuario
    st.write("Ingrese los datos de entrada:")

    age = st.number_input("Edad", min_value=0, max_value=100)
    gender = st.selectbox("Sexo", ["Male", "Female", "Other"])
    education_level = st.selectbox("Nivel de educación", EDUCATION_LEVEL_ENCODING)
    job_title = st.selectbox("Nombre del trabajo", JOB_TITLE_ENCODING)
    years_of_experience = st.number_input("Años de experiencia", min_value=0)
    seniority = st.selectbox("Tiempo de experiencia", SENIORITY_ENCODING)

    # Botones para hacer la predicción
    if st.button("Prediccion"):
        input_data = np.array([age, GENDER_ENCODING[gender], EDUCATION_LEVEL_ENCODING[education_level], JOB_TITLE_ENCODING[job_title], years_of_experience, SENIORITY_ENCODING[seniority]])
        input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        
        # Realizar la predicción del salario
        with torch.no_grad():
            output = model(input_data)

        print(f"Salida del modelo: {output}")

        # Mostrar la predicción
        st.write(f"El salario predicho es: {output}")
