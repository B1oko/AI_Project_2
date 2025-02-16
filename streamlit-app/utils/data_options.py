OPTIONS_GENDER = ["Male", "Female", "Other"]
OPTIONS_EDUCATION_LEVEL = ["Bachelor's", "Master's", "PhD", "High School"]

OPTIONS_JOB_CATEGORY = ['Regular', 'Senior', 'Junior', 'Director']

MAP_JOB_TYPE_JOB_TITLE = {
    'Software & IT': [
        'Software Engineer', 'Software Developer', 'Project Engineer',
        'Web Developer', 'Software Engineer Manager', 'Back end Developer',
        'Full Stack Engineer', 'Front end Developer', 'Front End Developer'
    ],
    'Data & Analytics': [
        'Data Analyst', 'Marketing Analyst', 'Financial Analyst',
        'Business Analyst', 'Data Scientist', 'Research Scientist',
        'Data Science'
    ],
    'Sales & Marketing': [
        'Sales Associate', 'Marketing Coordinator', 'Sales',
        'Sales Executive', 'Marketing', 'Sales Representative',
        'Digital Marketing Specialist'
    ],
    'Management': [
        'Product Manager', 'Sales Manager', 'Project Manager',
        'Operations Manager', 'Marketing Manager', 'Financial Manager',
        'Social Media Manager', 'Digital Marketing Manager', 'Content Marketing Manager',
        'Product Marketing Manager', 'Human Resources Manager'
    ],
    'Design & Creative': ['Product Designer', 'Graphic Designer'],
    'Human Resources': [
        'HR Generalist', 'HR Coordinator', 'Human Resources Coordinator', 'HR'
    ],
    'Operations & Logistics': ['Operations'],
    'Science & Research': ['Research'],
    'Other': ['Receptionist']
}

# # CODIGO PARA GENERAR MAP 
# agrupaciones = {}
# for typee, title in data[["Job Title", "Job Type"]].values:
#     if title not in agrupaciones:
#         agrupaciones[title] = [typee]
#     else:
#         if typee not in agrupaciones[title]:
#             agrupaciones[title].append(typee)