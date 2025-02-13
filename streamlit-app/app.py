import streamlit as st
from multiapp import MultiApp
from apps import eda, home, prediction

def main():

    app = MultiApp()

    st.title("Salary Prediction App")

    app.add_app("Home", home.app)
    app.add_app("EDA", eda.app)
    app.add_app("Prediction", prediction.app)
    app.run()

if __name__ == "__main__":
    main()