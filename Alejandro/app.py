import streamlit as st
from PIL import Image
 # import your app modules here
#from Prediction import pred
from EDA import eda


def main():

    Welcome = st.sidebar.button('Welcome')
    Eda = st.sidebar.button('Exploratory Data Analysis')
    Pred = st.sidebar.button('Prediction App')

    if Welcome:
        st.title("Salary Prediction App")
        st.write("Welcome to the Salary Prediction App!")
        img = Image.open("img/corpo.jpg")
        st.image(img)
        Eda = False
        Pred = False

    if Eda:
        eda()
        Welcome = False
        Pred = False

    if Pred:
        #pred()
        Welcome = False
        Eda = False
main()