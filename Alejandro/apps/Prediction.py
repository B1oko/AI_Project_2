import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def app():
    st.title('Predicciones')

    st.write("Aqui se permitira seleccionar variables para realizar una predicción")
