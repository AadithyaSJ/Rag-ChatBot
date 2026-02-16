import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    details = pd.read_csv("data/all_season_details.csv", low_memory=False)
    summary = pd.read_csv("data/all_season_summary.csv", low_memory=False)
    points = pd.read_csv("data/points_table.csv", low_memory=False)
    batting = pd.read_csv("data/all_season_batting_card.csv", low_memory=False)
    bowling = pd.read_csv("data/all_season_bowling_card.csv", low_memory=False)


    return details, summary, points, batting, bowling
