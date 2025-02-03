import streamlit as st
from multiapp import MultiApp
from apps import EDA, home, Prediction

app = MultiApp()

st.title("Salary Prediction App")

app.add_app("Home", home.app)
app.add_app("EDA", EDA.app)
app.add_app("Prediction", Prediction.app)
app.run()
