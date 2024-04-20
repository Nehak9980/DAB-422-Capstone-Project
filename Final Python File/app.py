import streamlit as st
import pandas as pd
import joblib

# Load the policymaker model pipeline
model_pipeline_pm = joblib.load('lasso_pipeline.joblib')

# Load the investor model pipeline
model_pipeline_inv = joblib.load('lasso_pipeline_investor.joblib')

# Load the master data
master_data_path = 'master_data.xlsx'
master_data = pd.read_excel(master_data_path)

# Define features for each user type
policy_features = ['Province', 'Paralegals and Social Service Workers', 'Inflation Rate', 'Mortgage Rate', 
                    'Temp Resident to Permanent Resident', 'Population', 'Resettled Refugee', 
                    'Worker Program', 'Construction and Equipment Operations', 'Vacancy Rate', 
                    'COVID Indicator', 'Business Immigration', 'Construction and Facility Management', 
                    'Transportation and Natural Resources Management']

investor_features = ['Province','Inflation Rate', 'Mortgage Rate', 'Vacancy Rate']

def user_input_features(features, context):
    inputs = {}
    # Categorical feature input
    if 'Province' in features:
        inputs['Province'] = st.selectbox('Province', master_data['Province'].unique(), key=f'{context}_Province_selectbox')
    
    # Numeric features input
    for feature in features:
        if feature != 'Province':  # Exclude Province from the numeric inputs
            # Generate a unique key for each input using the feature name and context
            key = f"{context}_{feature.replace(' ', '_').lower()}"  # Create a unique key for each feature
            inputs[feature] = st.number_input(feature, value=0, key=key)
    
    return pd.DataFrame([inputs])

def main():
    st.title('HPI Prediction Tool')
    
    st.image("canada_image1.jpg")

    # Custom CSS to inject into the webpage to display the background image
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("canada_image1.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["Policy Maker View", "Investor View"])
    
    with tab1:
        st.write("Policy Maker Dashboard")
        input_pm = None  # Initialize input_df to None
        input_pm = user_input_features(policy_features, 'policymaker')
        if st.button('Predict HPI for Policymakers', key='predict_policymaker') and input_pm is not None:
            prediction = model_pipeline_pm.predict(input_pm)
            st.write(f'Predicted HPI: {prediction[0]}')

    with tab2:
        st.write("Real Estate Investor Dashboard")
        input_inv = None  # Initialize input_df to None
        input_inv = user_input_features(investor_features, 'investor')
        if st.button('Predict HPI for Investors', key='predict_investor') and input_inv is not None:
            prediction = model_pipeline_inv.predict(input_inv)
            st.write(f'Predicted HPI: {prediction[0]}')

if __name__ == '__main__':
    main()