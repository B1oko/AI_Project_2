import streamlit as st
import pandas as pd
import numpy as np
#lets create our first static visualizations using Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    data = pd.read_csv('data/salary_data_cleaned_small.csv')
    job_title_counts = data['Job Title'].value_counts()
    job_titles_to_keep = job_title_counts[job_title_counts >= 40].index
    data = data[data['Job Title'].isin(job_titles_to_keep)]
    return data


def app():
    st.title("Exploratory Data Analysis")
    st.markdown("---")
    # First we need to load our data

    data = load_data()

    def display_random(data):
        sample = data.sample(6)
        return sample

    # Descripcion del Dataset
    st.subheader('El Dataset')
    st.write(
        "Nuestro dataset consta de un total de 6522 registros y 7 columnas, algunas de ellas son categoricas y otras no:\n"
        "| Variable            | Descripción |\n"
        "|---------------------|-------------|\n"
        "| **Age** | Edad del en años. |\n"
        "| **Gender** | Género (\"Male\" para masculino, \"Female\" para femenino). |\n"
        "| **Education Level** | Nivel educativo alcanzado (Grado, Máster, PhD...).|\n"
        "| **Job Title** | Nombre del empleo (Software Engineer, Marketing Manager, ...). |\n"
        "| **Years of Experience** | Años de experiencia laboral en el campo o industria. |\n"
        "| **Salary** | Salario anual del empleado en dólares, expresado con separadores de miles. |\n"
        "| **Job Category** | Categoría del puesto (Junior, Senior, ...). |\n"
        "| **Job Type** | Tipo de trabajo o industria del puesto (Management, Software & IT, ...). |\n"
        '\n\n'
        "Pulsa en el siguiente botón para ver 6 filas aleatorias del dataset"
    )

    new_button = st.button('Display 6 random rows')
    if new_button:
        sample = display_random(data)
        st.dataframe(sample)
    
    st.write("<br>", unsafe_allow_html=True)

    # Plot de las variables
    st.subheader('Descripción de las Características')
    
    categorical_columns = ['Gender', 'Education Level', 'Job Category']
    num_var = ['Age', 'Years of Experience', 'Salary']
    
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        var = st.radio('', ('Salario', 'Edad', 'Años de Experiencia', 'Género', 'Nivel de Educación', 'Categoría del Trabajo', 'Título del Trabajo'))
    with col2:
        with st.container():
            # if var in num_var:
            #     # st.subheader(f'Histogram of {var}')
            #     pass
            # elif var in categorical_columns:
            #     value_count=data[var].value_counts()
            #     col1,col2=st.columns(2)

            #     # st.subheader('Pie Chart')
            #     fig,ax=plt.subplots()
            #     ax.pie(value_count,autopct='%0.2f%%',labels=value_count.index)
            #     st.pyplot(fig, transparent=True)

            #     # st.subheader('Bar Chart')
            #     value_count = data[var].value_counts()
            #     colors = sns.color_palette('flare', len(value_count))
            #     fig, ax = plt.subplots()

            #     ax.bar(value_count.index, value_count.values, color=colors)
            #     ax.set_ylabel('Count')
            #     ax.set_xticklabels(value_count.index, rotation=70)
            #     ax.set_xlabel(var)
            #     st.pyplot(fig, transparent=True)

            # else:
            #     value_count = data[var].value_counts()
            #     colors = sns.color_palette('flare', len(value_count))
            #     fig, ax = plt.subplots()
            #     ax.bar(value_count.index, value_count.values, color=colors)
            #     ax.set_xticklabels(value_count.index, rotation=90)
            #     ax.set_xlabel(var)
            #     ax.set_ylabel('Count')
            #     st.pyplot(fig, transparent=True)

            if var == 'Salario':

                fig,ax=plt.subplots()
                ax.hist(data["Salary"], bins=30, edgecolor='k', color='#00ffa6')
                ax.set_xlabel("Salario")
                ax.set_ylabel("Frecuencia")

                # Colors
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                
                st.pyplot(fig, transparent=True)

                st.write(
                    "El salario representa la cantidad de dinero (En este caso en dólares) que un trabajador recibe a cambio de su trabajo."
                    "En este estudio es nuestra variable objetivo, es decir, la variable que queremos predecir mediante el resto."
                )
            
            elif var == 'Edad':

                fig,ax=plt.subplots()
                # ax.hist(data["Age"], bins=30, edgecolor='k', color='#00ffa6')
                sns.histplot(data["Age"], bins=40, kde=True, color='#00ffa6')
                ax.set_xlabel("Edad")
                ax.set_ylabel("Frecuencia")

                # Colors
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                
                st.pyplot(fig, transparent=True)

                st.write(
                    "La edad está obviamente medida en años y es una variable continua."
                    " Es una variable importante para el salario, puesto que la gente suele cobrar más cuanto más mayor y más experiencia tienen."
                )
                
            elif var == 'Años de Experiencia':

                fig,ax=plt.subplots()
                sns.histplot(data["Years of Experience"], bins=30, kde=True, color='#00ffa6')
                ax.set_xlabel("Años de experiencia")
                ax.set_ylabel("Frecuencia")

                # Colors
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                
                st.pyplot(fig, transparent=True)

                st.write(
                    "También obviamente medida en años y continua. Es una variable muy estrechamente relacionada con la primera por los mismos motivos."
                )

            elif var == 'Género':

                # Crear un gráfico de pastel
                fig,ax=plt.subplots()
                ax.pie(data["Gender"].value_counts(), labels=data["Gender"].value_counts().index, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', 
                       colors=["#00ffa6", "#003e3d", "#40bf93"], wedgeprops=dict(edgecolor='black'), textprops={'color': 'white'})

                st.pyplot(fig, transparent=True)

                st.write(
                    "El género es una variable categórica que indica si el trabajador es hombre, mujer u otro. Es una variable importante para el salario, puesto que existe una brecha salarial entre hombres y mujeres."
                )

            elif var == 'Nivel de Educación':
                var_count=data["Education Level"].value_counts()
                fig,ax=plt.subplots()
                ax.pie(var_count, labels=var_count.index, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', 
                       colors=["#00ffa6", "#003e3d", "#40bf93", "#708f84"], wedgeprops=dict(edgecolor='black'), textprops={'color': 'white'})

                st.pyplot(fig, transparent=True)

                st.write(
                        "El nivel de educación es una variable categórica que indica el nivel de estudios del trabajador. Es una variable importante para el salario, puesto que a mayor nivel de estudios, mayor salario o por lo menos mejor el puesto o empleo al que se puede acceder."
                    )
                
            elif var == 'Categoría del Trabajo':

                var_count=data["Job Category"].value_counts()
                fig,ax=plt.subplots()
                ax.pie(var_count, labels=var_count.index, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', 
                       colors=["#00ffa6", "#003e3d", "#40bf93", "#708f84"], wedgeprops=dict(edgecolor='black'), textprops={'color': 'white'})

                st.pyplot(fig, transparent=True)

                st.write(
                        "En este caso esta variable específica el rango que ocupa el empleado dentro de su puesto. Junior si en nuevo en el puesto y tiene poca experiencia. Regular si ya ha pasado el periodo de junior y, por lo tanto, suele tener mayores responsabilidades y salario. Senior para aquellos empleados con mucha experiencia en su empleo y que suelen requerir una mejora salarial. Y por último, director como persona encargada de gestionar a las tareas y obligaciones de los empleados de su departamento de trabajo.  Cuantas mayores son las responsabilidades y la experiencia requerida para alcanzar una posición mayor suele ser el salario asociado. "
                    )
                
            elif var == 'Título del Trabajo':

                var_count = data["Job Title"].value_counts()
                filtered_job_titles = var_count[var_count > 10]

                # Crear el gráfico de barras
                fig,ax=plt.subplots(figsize=(8, 15))
                sns.barplot(x=filtered_job_titles.values, y=filtered_job_titles.index, color='#00ffa6')

                # Colors
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')

                st.pyplot(fig, transparent=True)

                st.write(
                    "Por último el nombre del empleo, esta variable sirve para especificar la división del negocio a la que se dedica el empleado. No todos los trabajos son iguales, algunos requieren mayores estudios o experiencia para llevarse a cabo y la reducción en posibles empleados cualificados conlleva un aumento salarial comparado con otros empleos.  En este caso para que la gráfica sea legible se ha optado por solo graficar los 40 empleos más comunes en la base de datos. "
                )

if __name__ == "__main__":
    app()