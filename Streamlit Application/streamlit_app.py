import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

# Constants
STATE_OPTIONS = ['CA', 'NY', 'MA', 'TX', 'WA', 'other']
CATEGORY_OPTIONS = ['software', 'web', 'mobile', 'enterprise', 'advertising', 'games_video', 'semiconductor', 'network_hosting', 'biotech', 'hardware', 'ecommerce', 'public_relations', 'other']
CITY_OPTIONS = ['San Francisco', 'New York', 'Mountain View', 'Palo Alto', 'Santa Clara', 'other']

@st.cache_data
def load_data():
    return pd.read_csv('startup_data.csv')

def preprocess_data(df):
    df = df.drop(['Unnamed: 0', 'Unnamed: 6', 'latitude', 'longitude', 'zip_code', 'id', 'name', 'object_id'], axis=1)
    
    df['State'] = df['state_code'].map(lambda x: x if x in STATE_OPTIONS[:5] else 'other')
    df['category'] = df['category_code'].map(lambda x: x if x in CATEGORY_OPTIONS[:-1] else 'other')
    df['City'] = df['city'].map(lambda x: x if x in CITY_OPTIONS[:-1] else 'other')
    
    df = df.drop(['state_code', 'state_code.1', 'is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate',
                  'city', 'labels', 'category_code', 'is_software', 'is_web', 'is_mobile', 'is_enterprise',
                  'is_advertising', 'is_gamesvideo', 'is_ecommerce', 'is_biotech', 'is_consulting',
                  'is_othercategory'], axis=1)
    
    df['founded_year'] = pd.to_datetime(df['founded_at']).dt.year
    df = df[~((df['closed_at'].notna()) & (df['status'] == 'acquired'))]
    df = df.drop(['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at'], axis=1)
    
    for col in ['age_first_funding_year', 'age_last_funding_year', 'age_first_milestone_year', 'age_last_milestone_year']:
        df[col] = df[col].clip(lower=0)
        if col.startswith('age_first_milestone') or col.startswith('age_last_milestone'):
            df[col] = df[col].fillna(df[col].mode()[0])
    
    df['has_RoundABCD'] = ((df[['has_roundA', 'has_roundB', 'has_roundC', 'has_roundD']] == 1).any(axis=1)).astype(int)
    df['has_Investor'] = ((df[['has_VC', 'has_angel']] == 1).any(axis=1)).astype(int)
    df['has_Seed'] = ((df['has_RoundABCD'] == 0) & (df['has_Investor'] == 1)).astype(int)
    df['invalid_startup'] = ((df['has_RoundABCD'] == 0) & (df['has_VC'] == 0) & (df['has_angel'] == 0)).astype(int)
    
    df = pd.get_dummies(df, columns=['State', 'category', 'City'])
    df['status'] = (df['status'] == 'acquired').astype(int)
    
    return df.drop('status', axis=1), df['status']

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def main():
    st.title('Startup Success Predictor')
    data = load_data()
    X, y = preprocess_data(data)
    model, scaler = train_model(X, y)

    st.header('Enter Startup Information')
    input_data = {}
    input_data['company_name'] = st.text_input('Company Name')
    input_data['foundation_date'] = st.date_input('Foundation Date')
    input_data['age_first_funding_year'] = st.number_input('Age at First Funding (years)', min_value=0.0)
    input_data['age_last_funding_year'] = st.number_input('Age at Last Funding (years)', min_value=0.0)
    input_data['relationships'] = st.number_input('Number of Relationships', min_value=0)
    input_data['funding_rounds'] = st.number_input('Number of Funding Rounds', min_value=0)
    input_data['funding_total_usd'] = st.number_input('Total Funding (USD)', min_value=0)
    input_data['milestones'] = st.number_input('Number of Milestones', min_value=0)
    input_data['state'] = st.selectbox('State', options=STATE_OPTIONS)
    input_data['category'] = st.selectbox('Category', options=CATEGORY_OPTIONS)
    input_data['city'] = st.selectbox('City', options=CITY_OPTIONS)
    input_data['has_VC'] = st.checkbox('Has Venture Capital Funding')
    input_data['has_angel'] = st.checkbox('Has Angel Funding')
    input_data['has_roundA'] = st.checkbox('Has Round A Funding')
    input_data['has_roundB'] = st.checkbox('Has Round B Funding')
    input_data['has_roundC'] = st.checkbox('Has Round C Funding')
    input_data['has_roundD'] = st.checkbox('Has Round D Funding')
    input_data['avg_participants'] = st.number_input('Average Number of Funding Participants', min_value=0.0)
    input_data['is_top500'] = st.checkbox('Is in Top 500')

    if st.button('Predict Success'):
        prediction_input = prepare_input_data(input_data, X.columns)
        prediction_input_scaled = scaler.transform(prediction_input)
        probability = model.predict_proba(prediction_input_scaled)[0][1]
        st.subheader('Prediction Result')
        st.success(f'The Probability of success: {probability:.2%}')

def prepare_input_data(input_data, columns):
    df = pd.DataFrame(input_data, index=[0])
    df['has_RoundABCD'] = ((df[['has_roundA', 'has_roundB', 'has_roundC', 'has_roundD']] == 1).any(axis=1)).astype(int)
    df['has_Investor'] = ((df[['has_VC', 'has_angel']] == 1).any(axis=1)).astype(int)
    df['has_Seed'] = ((df['has_RoundABCD'] == 0) & (df['has_Investor'] == 1)).astype(int)
    df['invalid_startup'] = ((df['has_RoundABCD'] == 0) & (df['has_VC'] == 0) & (df['has_angel'] == 0)).astype(int)
    
    df = pd.get_dummies(df, columns=['state', 'category', 'city'], prefix=['State', 'category', 'City'])
    
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    
    return df[columns]

if __name__ == "__main__":
    main()