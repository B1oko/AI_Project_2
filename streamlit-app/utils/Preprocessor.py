import pickle
import os
import pandas as pd

class Preprocessor():
    def __init__(self):
        with open("models/preprocessing_pipeline.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)

    def transform(self, data: list):
        inferencia_df = pd.DataFrame([data], columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Job Category", "Job Type"])
        return self.preprocessor.transform(inferencia_df)



def preprocesar_inferencia(inferencia):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "models", "preprocessing_pipeline.pkl")
    
    with open(model_path, "rb") as f:
        preprocessor = pickle.load(f)
    
    inferencia_df = pd.DataFrame([inferencia], columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Job Category", "Job Type"])
    inferencia_transformed = preprocessor.transform(inferencia_df)

    return inferencia_transformed

