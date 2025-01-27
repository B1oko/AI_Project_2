import streamlit as st
import pandas as pd
import numpy as np

#lets create our first static visualizations using Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

st.header("EDA")

#First we need to load our data

data = pd.read_csv('../Data/Salary_Data.csv')
data['Education Level'] = data['Education Level'].replace('phD', 'PhD')

#Create a funcion to randomly show us 3 rows of a dataframe
def display_random(data):
	sample=data.sample(6)
	return sample

st.markdown("---")

#we create the button to submit the function
st.subheader('Displaying 6 rows')
st.caption('click on the button to display')
new_button=st.button('Display 6 random rows')
if new_button:
	sample = display_random(data)
	st.dataframe(sample)


categorical_columns = ['Gender', 'Education Level']

st.subheader('Choose a variable to plot')
var=st.radio('Pick one',
	('Salary','Age','Years of Experience', 'Gender', 'Education Level','Job Title'))
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
	
		with col1:
			st.subheader('Pie Chart')
			fig,ax=plt.subplots()
			ax.pie(value_count,autopct='%0.2f%%',labels=data[var].unique())
			st.pyplot(fig)

		with col2:
			st.subheader('Bar Chart')
			fig,ax=plt.subplots()
			ax.bar(value_count.index,value_count)
			st.pyplot(fig)
	else:
		fig,ax=plt.subplots()
		ax.bar(data["Job Title"],bins=40,edgecolor='k')
		ax.set_title('Barplot of Job Title')
		ax.set_xlabel("Job Title")
		ax.set_ylabel('Frequency')
		st.pyplot(fig)


st.markdown('---')

## SIN TOCAR

