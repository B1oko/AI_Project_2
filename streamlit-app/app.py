import streamlit as st
from multiapp import MultiApp
from apps import eda, home, prediction

def main():

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    local_css("style.css")

    app = MultiApp()

    app.add_app("Home", home.app)
    app.add_app("EDA", eda.app)
    app.add_app("Prediction", prediction.app)
    app.run()

    st.sidebar.markdown("""
        <div class="sidebar-footer">
            <p class="sidebar-footer-subheading">Desarrollado por:</p>
            <p><a href="https://github.com/B1oko" target="_blank">Pablo Esteban</a></p>
            <p><a href="https://github.com/paujorques02" target="_blank">Pau Jorques</a></p>
            <p><a href="https://github.com/VeintimillAlejandro" target="_blank">Alejandro Veintimilla</a></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()