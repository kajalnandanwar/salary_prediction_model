
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("💼 Data Science Salary Predictor")

experience = st.selectbox("Experience Level", [0,1,2,3])
employment = st.selectbox("Employment Type", [0,1,2,3])
job = st.number_input("Job Title (encoded)")
residence = st.number_input("Employee Residence (encoded)")
remote = st.slider("Remote Ratio", 0, 100)
location = st.number_input("Company Location (encoded)")
company_size = st.selectbox("Company Size", [0,1,2])

if st.button("Predict Salary"):
    input_data = np.array([[experience, employment, job,
                            residence, remote,
                            location, company_size]])
    
    prediction = model.predict(input_data)
    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
