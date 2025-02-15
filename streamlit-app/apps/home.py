import streamlit as st
from PIL import Image

def app():
    st.title("Salary Prediction App")
    st.write("Bienvenido al Salary Prediction App!")
    st.write(
        "Descubre cómo tu potencial salarial puede superar todas las expectativas con nuestra innovadora aplicación de predicción de sueldos. "
        "Utilizamos tecnología de vanguardia en Machine Learning y redes neuronales para analizar, en cuestión de segundos, factores clave como edad, experiencia profesional, formación o tipo de puesto.\n\n"
        "Gracias a estos potentes algoritmos, nuestro sistema identifica patrones y estima rangos de salarios precisos y realistas, ofreciéndote una referencia invaluable para la toma de decisiones en tu carrera profesional.\n\n"
        "Con esta herramienta podrás:\n\n"
        "- Comparar tu posición en el mercado y evaluar si tu remuneración actual está en línea con el promedio.\n"
        "- Planificar tu trayectoria laboral, identificando cuál puede ser el impacto de adquirir nuevas habilidades o cambiar de sector.\n"
        "- Negociar con mayor seguridad en procesos de selección, respaldado por predicciones sólidas basadas en datos reales.\n\n"
        "Haz que cada paso en tu desarrollo profesional cuente. ¡Únete a nuestro futuro compartido, impulsado por la inteligencia artificial, y conoce hoy mismo el salario que podrías estar ganando mañana!"
    )
    img = Image.open("./img/image.png")
    st.image(img)