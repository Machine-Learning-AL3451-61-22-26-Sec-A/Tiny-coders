import numpy as np
import pandas as pd
import streamlit as st
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

def main():
    st.title("Bayesian Network for Heart Disease Prediction")

    # File uploader to load the CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Load the dataset
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
        heartDisease = pd.read_csv(uploaded_file, names=names)
        heartDisease = heartDisease.replace('?', np.nan)  # Replace '?' with NaN
        st.write('Dataset preview:', heartDisease.head())

        # Define the Bayesian model structure
        model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'),
                               ('exang', 'trestbps'), ('trestbps', 'heartdisease'),
                               ('fbs', 'heartdisease'), ('heartdisease', 'restecg'),
                               ('heartdisease', 'thalach'), ('heartdisease', 'chol')])

        # Fit the model using Maximum Likelihood Estimator
        model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

        # Input fields for evidence
        age = st.number_input('Age', min_value=1, max_value=100, value=37)
        sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')

        # Perform inference
        HeartDisease_infer = VariableElimination(model)
        q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': age, 'sex': sex})

        # Print the query result
        st.write('Inference result:')
        st.write(q['heartdisease'])

if __name__ == "__main__":
    main()
