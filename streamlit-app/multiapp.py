import streamlit as st
from PIL import Image

class MultiApp:
    """Framework for combining multiple Streamlit applications."""

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application."""
        self.apps.append({"title": title, "function": func})

    def run(self):
        img = Image.open("./img/logo3.png")

        # Mostrar la imagen en la parte superior de la barra lateral
        st.sidebar.image(img)

        # Aplicar el menú mejorado en la barra lateral
        app = st.sidebar.radio(
            '',
            self.apps,
            format_func=lambda app: app['title']
        )

        # Ejecutar la función de la página seleccionada
        app['function']()