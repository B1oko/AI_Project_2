import streamlit as st
from multiapp import MultiApp
from apps import eda, home, prediction

def main():

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    local_css("styles/style.css")

    app = MultiApp()

    app.add_app("Home", home.app)
    app.add_app("EDA", eda.app)
    app.add_app("Prediction", prediction.app)
    app.run()

if __name__ == "__main__":
    main()