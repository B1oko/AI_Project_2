import streamlit as st
import pandas as pd
import numpy as np
#lets create our first static visualizations using Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def app():
	st.title("Exploratory Data Analysis")
	st.markdown("---")
	#First we need to load our data

	data = pd.read_csv('../Data/salary_data_pau.csv')
	"""
	data['Education Level'] = data['Education Level'].replace('phD', 'PhD')
	job_title_counts = data['Job Title'].value_counts()
	data['Job Title'] = data['Job Title'].astype(str)

	def categorize_job_title(title):
		if title.startswith('Senior'):
			return 'Senior'
		elif title.startswith('Junior'):
			return 'Junior'
		else:
			return 'Regular'

	data['Job Category'] = data['Job Title'].apply(categorize_job_title)
	data['Job Title'] = data['Job Title'].str.replace('Senior', '').str.replace('Junior', '').str.strip()
	print(data[['Job Title', 'Job Category']].head())

	job_titles_to_keep = job_title_counts[job_title_counts >= 40].index
	data = data[data['Job Title'].isin(job_titles_to_keep)]
	"""

	#Create a funcion to randomly show us 3 rows of a dataframe
	def display_random(data):
		sample=data.sample(6)
		return sample

	#we create the button to submit the function
	st.subheader('Displaying 6 rows')
	st.caption('click on the button to display')
	new_button=st.button('Display 6 random rows')
	if new_button:
		sample = display_random(data)
		st.dataframe(sample)


	categorical_columns = ['Gender', 'Education Level','Job Category',"Job Type"]

	st.subheader('Choose a variable to plot')
	var=st.radio('Pick one',
		('Salary','Age','Years of Experience', 'Gender', 'Education Level','Job Category','Job Type'))
	num_var = ['Age', 'Years of Experience', 'Salary']

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

		"""else:
			value_count = data[var].value_counts()
			colors = sns.color_palette('flare', len(value_count))
			fig, ax = plt.subplots()
			ax.bar(value_count.index, value_count.values, color=colors)
			ax.set_xticklabels(value_count.index, rotation=90)
			ax.set_xlabel(var)
			ax.set_ylabel('Count')
			st.pyplot(fig)"""
