import streamlit as st
from multiapp import MultiApp
from apps import eda, home, prediction, modeling

def main():

    app = MultiApp()

    st.title("Salary Prediction App")

    app.add_app("Home", home.app)
    app.add_app("EDA", eda.app)
    app.add_app("Prediction", prediction.app)
    app.add_app("Modeling", modeling.app)
    app.run()

if __name__ == "__main__":
    main()