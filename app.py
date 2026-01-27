import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="KL Dining Assistant",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM CSS (THEME: #355C7D & #C06C84)
# ==========================================
st.markdown("""
<style>
    /* 1. MAIN BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #355C7D 0%, #6C5B7B 50%, #C06C84 100%);
        color: #FFFFFF;
    }

    /* 2. HEADINGS */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* 3. METRIC CARDS */
    div[data-testid="stMetricValue"] {
        color: #F8B195 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }

    /* 4. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #2A3E50;
        color: #FFFFFF;
    }
    
    /* 5. BUTTONS */
    div.stButton > button {
        background-color: #F8B195;
        color: #355C7D;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
    }
    div.stButton > button:hover {
        background-color: #C06C84;
        color: white;
    }

    /* 6. TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(0,0,0,0.3);
        color: #FFFFFF;
        border-radius: 5px 5px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F8B195;
        color: #355C7D;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    filename = 'streamlitdata.csv'
    if not os.path.exists(filename):
        return None
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        return None

df = load_data()

if df is None:
    st.error("Error: 'streamlitdata.csv' not found. Please ensure the data file is in the same directory.")
    st.stop()

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("KL Dining Assistant")
st.sidebar.markdown("By Tanisya Pristi Azrelia")
st.sidebar.caption("Master in Data Science - Universiti Malaya")
