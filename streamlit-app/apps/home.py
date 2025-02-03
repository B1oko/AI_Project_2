import streamlit as st
from PIL import Image

def app():
    st.write("Welcome to the Salary Prediction App!")
    img = Image.open("./img/corpo.jpg")
    st.image(img)