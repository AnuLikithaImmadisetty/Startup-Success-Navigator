import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from streamlit_option_menu import option_menu

# Constants
STATE_OPTIONS = ['CA', 'NY', 'MA', 'TX', 'WA', 'other']
CATEGORY_OPTIONS = ['software', 'web', 'mobile', 'enterprise', 'advertising', 'games_video', 'semiconductor', 'network_hosting', 'biotech', 'hardware', 'ecommerce', 'public_relations', 'other']
CITY_OPTIONS = ['San Francisco', 'New York', 'Mountain View', 'Palo Alto', 'Santa Clara', 'other']

# Set page configuration
st.set_page_config(
    page_title="Startup Success Navigator🚀",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded",
)

# Sidebar for navigation
with st.sidebar:
    options = ['Home', 'Startup Success Prediction', 'About the Model']
    selected = option_menu('Startup Success Navigator🚀',
                           options,
                           menu_icon='graph-up-arrow',
                           icons=['house-door', 'briefcase-fill', 'info-circle'],
                           default_index=0)

# Load and preprocess data
@st.cache_data
def load_data():
    return pd.read_csv('startup_data.csv')

def preprocess_data(df):
    # Drop unnecessary columns, ignoring errors for non-existing columns
    df = df.drop(['Unnamed: 0', 'Unnamed: 6', 'latitude', 'longitude', 'zip_code', 'id', 'name', 'object_id'], axis=1, errors='ignore')

    # Check if 'state_code' exists before mapping
    if 'state_code' in df.columns:
        df['State'] = df['state_code'].map(lambda x: x if x in STATE_OPTIONS[:5] else 'other')
    else:
        df['State'] = 'other'  # Assign 'other' if 'state_code' is missing

    # Check if 'category_code' exists before mapping
    if 'category_code' in df.columns:
        df['category'] = df['category_code'].map(lambda x: x if x in CATEGORY_OPTIONS[:-1] else 'other')
    
    # Check if 'city' exists before mapping
    if 'city' in df.columns:
        df['City'] = df['city'].map(lambda x: x if x in CITY_OPTIONS[:-1] else 'other')

    # Drop other unnecessary columns, ignoring non-existing columns
    df = df.drop(['state_code', 'state_code.1', 'is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate',
                  'city', 'labels', 'category_code', 'is_software', 'is_web', 'is_mobile', 'is_enterprise',
                  'is_advertising', 'is_gamesvideo', 'is_ecommerce', 'is_biotech', 'is_consulting',
                  'is_othercategory'], axis=1, errors='ignore')

    # Extract year from 'founded_at' if the column exists
    if 'founded_at' in df.columns:
        df['founded_year'] = pd.to_datetime(df['founded_at']).dt.year

    # Remove rows where status is 'acquired' but 'closed_at' is not NaN
    if 'closed_at' in df.columns and 'status' in df.columns:
        df = df[~((df['closed_at'].notna()) & (df['status'] == 'acquired'))]

    # Drop date-related columns that are no longer needed
    df = df.drop(['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at'], axis=1, errors='ignore')
    
    return df

# Function to train and save the model
def train_model(df):
    # Preprocess the data
    df = preprocess_data(df)
    
    # Features and target
    X = df.drop(columns=['status'])
    y = df['status'].apply(lambda x: 1 if x == 'acquired' else 0)  # 1 for 'acquired', 0 for 'closed'
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numeric features (useful if you have numeric columns to scale)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = XGBClassifier()
    model.fit(X_train_scaled, y_train)
    
    # Save the model
    with open('models/startup_success_model.sav', 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Function to load the model or train if not found
def load_or_train_model(df):
    model_path = 'models/startup_success_model.sav'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = train_model(df)
    return model

# Home Page
if selected == 'Home':
    st.title('Welcome to the Startup Success Navigator🚀')
    
    st.markdown("""
    ## Predicting Startup Success with AI
    
    **Startup Success Navigator** is designed to help you predict the likelihood of a startup being acquired or closed based on historical data. 
    Navigate to the **Startup Success Prediction** section to input startup data and get predictions.
    
    ### Features:
    - Predict the success of startups based on factors like funding rounds, relationships, industries, and more.
    - Leverages advanced machine learning models for accurate predictions.
    - Pre-built with a rich dataset of real-world startups.
    
    Use the navigation on the left to explore the options.
    """)

# Startup Success Prediction Page
if selected == 'Startup Success Prediction':
    st.title('Startup Success Prediction')
    
    # Load dataset
    df = load_data()
    
    # Columns for user input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        state_input = st.selectbox('Select State', STATE_OPTIONS)
    with col2:
        category_input = st.selectbox('Select Category', CATEGORY_OPTIONS)
    with col3:
        city_input = st.selectbox('Select City', CITY_OPTIONS)
    
    funding_rounds = st.slider('Number of Funding Rounds', 0, 10, 1)
    relationships = st.slider('Number of Relationships', 0, 100, 1)
    
    # Prediction Button
    if st.button('Predict Success'):
        # Preprocess input
        user_data = pd.DataFrame({
            'State': [state_input],
            'category': [category_input],
            'City': [city_input],
            'funding_rounds': [funding_rounds],
            'relationships': [relationships]
        })
        
        # Preprocessing pipeline for input data
        model_input = preprocess_data(user_data)
        
        # Load or train the model
        success_model = load_or_train_model(df)
        
        # Perform prediction
        prediction = success_model.predict(model_input)
        prediction_result = 'Acquired' if prediction[0] == 1 else 'Closed'
        
        st.success(f'The startup is predicted to be: **{prediction_result}**')

# About the Model
if selected == 'About the Model':
    st.title('About the Startup Success Model')
    
    st.markdown("""
    This project leverages a **XGBoost Classifier** trained on historical data of startups to predict whether a startup will be acquired or closed. 
    
    ### Model Overview:
    - **Algorithm**: XGBoost Classifier
    - **Preprocessing**: Robust Scaling for numerical data
    - **Features**: Includes variables like funding rounds, relationships, and industry type.
    
    ### How It Works:
    The machine learning model uses various startup attributes to make predictions. The attributes include:
    - Location (State, City)
    - Industry Category
    - Number of Funding Rounds
    - Number of Relationships
    """)
    
    st.write("To learn more or contribute, visit the [Startup-Success-Navigator](https://github.com/AnuLikithaImmadisetty/Startup-Success-Navigator).")
