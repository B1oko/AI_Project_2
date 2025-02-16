import streamlit as st
import pandas as pd
import numpy as np
#lets create our first static visualizations using Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
	data = pd.read_csv('Notebooks/data/salary_data_pau_cleaned_small_modified.csv')
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

	st.subheader('Displaying 6 rows')
	st.caption('Click the button to display random rows')
	new_button = st.button('Display 6 random rows')
	if new_button:
		sample = display_random(data)
		st.dataframe(sample)

	categorical_columns = ['Gender', 'Education Level', 'Job Category']
	num_var = ['Age', 'Years of Experience', 'Salary']

	st.subheader('Choose a variable to plot')
	var = st.radio('Pick one', ('Salary', 'Age', 'Years of Experience', 'Gender', 'Education Level', 'Job Category', 'Job Title'))

	with st.container():
		if var in num_var:
			st.subheader(f'Histogram of {var}')
			fig,ax=plt.subplots()
			ax.hist(data[var],bins=40,edgecolor='k')
			ax.set_xlabel(var)
			ax.set_ylabel('Frequency')
			st.pyplot(fig)

		elif var in categorical_columns:
			value_count=data[var].value_counts()
			col1,col2=st.columns(2)

			st.subheader('Pie Chart')
			fig,ax=plt.subplots()
			ax.pie(value_count,autopct='%0.2f%%',labels=value_count.index)
			st.pyplot(fig)

			st.subheader('Bar Chart')
			value_count = data[var].value_counts()
			colors = sns.color_palette('flare', len(value_count))
			fig, ax = plt.subplots()
			ax.bar(value_count.index, value_count.values, color=colors)
			ax.set_ylabel('Count')
			ax.set_xticklabels(value_count.index, rotation=70)
			ax.set_xlabel(var)
			st.pyplot(fig)

		else:
			value_count = data[var].value_counts()
			colors = sns.color_palette('flare', len(value_count))
			fig, ax = plt.subplots()
			ax.bar(value_count.index, value_count.values, color=colors)
			ax.set_xticklabels(value_count.index, rotation=90)
			ax.set_xlabel(var)
			ax.set_ylabel('Count')
			st.pyplot(fig)

		if var == 'Salary':
			st.write(
        "El salario representa la cantidad de dinero (En este caso en dólares) que un trabajador recibe a cambio de su trabajo."
		" En este estudio es nuestra variable objetivo, es decir, la variable que queremos predecir mediante el resto."
	)
		elif var == 'Age':
			st.write(
        "La edad está obviamente medida en años y es una variable continua."
		" Es una variable importante para el salario, puesto que la gente suele cobrar más cuanto más mayor y más experiencia tienen."
	)
		elif var == 'Years of Experience':
			st.write(
        "También obviamente medida en años y continua. Es una variable muy estrechamente relacionada con la primera por los mismos motivos."
	)
		elif var == 'Gender':
			st.write(
        "El género es una variable categórica que indica si el trabajador es hombre, mujer u otro. Es una variable importante para el salario, puesto que existe una brecha salarial entre hombres y mujeres."
	)
		elif var == 'Education Level':
			st.write(
        "El nivel de educación es una variable categórica que indica el nivel de estudios del trabajador. Es una variable importante para el salario, puesto que a mayor nivel de estudios, mayor salario o por lo menos mejor el puesto o empleo al que se puede acceder."
	)
		elif var == 'Job Category':
			st.write(
        "En este caso esta variable específica el rango que ocupa el empleado dentro de su puesto. Junior si en nuevo en el puesto y tiene poca experiencia. Regular si ya ha pasado el periodo de junior y, por lo tanto, suele tener mayores responsabilidades y salario. Senior para aquellos empleados con mucha experiencia en su empleo y que suelen requerir una mejora salarial. Y por último, director como persona encargada de gestionar a las tareas y obligaciones de los empleados de su departamento de trabajo.  Cuantas mayores son las responsabilidades y la experiencia requerida para alcanzar una posición mayor suele ser el salario asociado. "
	)
		elif var == 'Job Title':
			st.write(
        "Por último el nombre del empleo, esta variable sirve para especificar la división del negocio a la que se dedica el empleado. No todos los trabajos son iguales, algunos requieren mayores estudios o experiencia para llevarse a cabo y la reducción en posibles empleados cualificados conlleva un aumento salarial comparado con otros empleos.  En este caso para que la gráfica sea legible se ha optado por solo graficar los 40 empleos más comunes en la base de datos. "
	)

if __name__ == "__main__":
	app()