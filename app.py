import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline
pipe = pickle.load(open('pip.pkl', 'rb'))

# Title
st.title("🏏 IPL Match Winner Predictor")

# Sidebar info
st.sidebar.header("Match Information")

# Input options
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Pune'
]

# User inputs
batting_team = st.selectbox('Select Batting Team', sorted(teams))
bowling_team = st.selectbox('Select Bowling Team', sorted(teams))
selected_city = st.selectbox('Select City', sorted(cities))
target = st.number_input('Target', min_value=0)
score = st.number_input('Current Score', min_value=0)
overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
wickets = st.number_input('Wickets Out', min_value=0, max_value=10, step=1)

# Feature engineering (same as training)
runs_left = target - score
balls_left = 120 - (overs * 6)
wickets_left = 10 - wickets
crr = score / overs if overs > 0 else 0
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

# DataFrame for prediction
input_df = pd.DataFrame({
    'batting_team':[batting_team],
    'bowling_team':[bowling_team],
    'city':[selected_city],
    'balls_left':[balls_left],
    'runs_left':[runs_left],
    'wicket_left':[wickets],   # ✅ FIXED NAME
    'total_runs_x':[target],
    'CRR':[crr],
    'RRR':[rrr]
})

# Prediction button
if st.button('Predict Probability'):
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    
    st.success(f"🏆 Winning Probability for {batting_team}: {round(win*100, 2)} %")
    st.error(f"💔 Losing Probability for {batting_team}: {round(loss*100, 2)} %")

