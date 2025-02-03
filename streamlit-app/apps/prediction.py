import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Definir el modelo de red neuronal
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



def app():
        
    # Interfaz de Streamlit
    st.title("Entrenamiento de Red Neuronal Simple")

    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Vista previa de los datos:", data.head())

        target_column = st.selectbox("Selecciona la variable objetivo", data.columns)

        if st.button("Entrenar Modelo"):
            # Preparar los datos
            X = data.drop(target_column, axis=1)
            y = data[target_column]

            # División en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Escalado de los datos
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Conversión a tensores de PyTorch
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

            # Inicializar el modelo, la función de pérdida y el optimizador
            model = SimpleNN(X_train.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Entrenamiento del modelo
            epochs = 100
            loss_history = []
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())

            # Evaluación del modelo
            model.eval()
            with torch.no_grad():
                predictions = model(X_test_tensor)
                test_loss = criterion(predictions, y_test_tensor)

            st.write(f"Pérdida final en el conjunto de prueba: {test_loss.item():.4f}")

            # Visualización de la pérdida durante el entrenamiento
            plt.figure()
            plt.plot(loss_history)
            plt.title('Historial de Pérdida durante el Entrenamiento')
            plt.xlabel('Épocas')
            plt.ylabel('Pérdida')
            st.pyplot(plt)
