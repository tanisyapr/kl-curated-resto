import streamlit as st
import pandas as pd
import plotly.express as px
import os

# =========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="KL Dining Assistant",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #355C7D 0%, #6C5B7B 50%, #C06C84 100%); color: #FFFFFF; }
    h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
    div[data-testid="stMetricValue"] { color: #F8B195 !important; }
    div[data-testid="stMetricLabel"] { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] { background-color: #2A3E50; color: #FFFFFF; }
    div.stButton > button { background-color: #F8B195; color: #355C7D; font-weight: bold; border: none; }
    div.stButton > button:hover { background-color: #C06C84; color: white; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. ROBUST DATA LOADING (FIXES ERRORS)
# ==========================================
@st.cache_data
def load_data():
    filename = 'streamlitdata.csv'
    if not os.path.exists(filename):
        return None
    try:
        df = pd.read_csv(filename)
        # CRITICAL FIX: Strip whitespace from column names to prevent KeyErrors
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        return None

df = load_data()

if df is None:
    st.error("CRITICAL ERROR: 'streamlitdata.csv' not found. Please upload it.")
    st.stop()

# ==========================================
# 4. SIDEBAR
# ==========================================
st.sidebar.title("KL Dining Assistant")
st.sidebar.markdown("By Tanisya Pristi Azrelia")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Best of The Best", "Find Your Restaurant", "Methodology & Insights"])

# DEBUG TOOL (Use this if it's still broken!)
if st.sidebar.checkbox("Show Data Columns (Debug)"):
    st.sidebar.write(list(df.columns))

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "Best of The Best":
    st.title("KL Restaurant Recommendation System")
    st.markdown("Welcome! Explore the top-rated establishments below.")
    st.divider()
    
    # Safety Check: Ensure columns exist before sorting
    if 'review_count' in df.columns and 'avg_rating' in df.columns:
        top_restaurants = df[df['review_count'] > 50].sort_values('avg_rating', ascending=False).head(20)
        
        st.dataframe(
            top_restaurants,
            column_order=["restaurant", "avg_rating", "review_count"],
            hide_index=True,
            use_container_width=True,
            height=600
        )
    else:
        st.error("Error: Dataset is missing 'review_count' or 'avg_rating' columns.")

# ==========================================
# PAGE 2: RECOMMENDATION ENGINE (FIXED)
# ==========================================
elif page == "Find Your Restaurant":
    st.title("Personalized Recommendation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 1. Select Preferences")
        
        # DYNAMIC OPTION LOADING
        # This prevents the app from crashing if a column name is slightly different.
        # We look for columns that match your topics.
        
        potential_topics = [
            "Food Quality", "Staff Friendliness", "Ambiance & Atmosphere", 
            "Service Operations/Speed", "Management", "Value for Money", "Cleanliness"
        ]
        
        # Only show options that ACTUALLY EXIST in the file
        available_options = [t for t in potential_topics if t in df.columns]
        
        if not available_options:
            st.error("No topic columns found in data! Check the Debug box in sidebar.")
            st.stop()
            
        priorities = st.multiselect(
            "What matters most?",
            options=available_options,
            default=[available_options[0]]
        )
        
        st.markdown("### 2. Cuisine Filter")
        cuisine_pref = st.radio("Category:", ["All Cuisines", "Western", "Asian"])
        
        st.markdown("---")
        btn = st.button("Find Best Place", type="primary")

    with col2:
        if btn:
            # 1. SCORING
            if priorities:
                # Calculate mean of selected columns
                df['final_score'] = df[priorities].mean(axis=1)
            else:
                # Fallback to stars if nothing selected
                if 'avg_rating' in df.columns:
                    df['final_score'] = df['avg_rating']
                else:
                    df['final_score'] = 0
            
            # 2. FILTERING
            filtered_df = df.copy()
            
            # Robust Column Checking for Cuisines
            # We use .str.contains to find columns even if named differently (e.g. "Western Cuisine" vs "Western")
            if cuisine_pref == "Western":
                # Find any column containing "Western"
                west_cols = [c for c in df.columns if "Western" in c]
                if west_cols:
                    filtered_df = filtered_df[filtered_df[west_cols[0]] > 3.0]
                    
            elif cuisine_pref == "Asian":
                # Find any column containing "Asian"
                asian_cols = [c for c in df.columns if "Asian" in c]
                if asian_cols:
                    filtered_df = filtered_df[filtered_df[asian_cols[0]] > 3.0]

            # 3. DISPLAY
            results = filtered_df.sort_values('final_score', ascending=False).head(5)
            
            if priorities:
                st.subheader(f"Top Recommendations for: {', '.join(priorities)}")
            else:
                st.subheader("Top Recommendations")

            if len(results) == 0:
                st.warning("No matches found. Try relaxing your filters.")
                
            for i, (index, row) in enumerate(results.iterrows()):
                with st.container():
                    st.markdown(f"### #{i+1} {row['restaurant']}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Match Score", f"{row['final_score']:.2f}")
                    
                    if 'avg_rating' in row:
                        c2.metric("Rating", f"{row['avg_rating']:.1f}")
                    
                    if 'review_count' in row:
                        c3.metric("Reviews", f"{int(row['review_count'])}")
                        
                    # Show the score of the first selected priority
                    if priorities:
                        first_p = priorities[0]
                        c4.metric(first_p, f"{row[first_p]:.1f}")
                    
                    st.markdown("---")

# ==========================================
# PAGE 3: INSIGHTS
# ==========================================
elif page == "Methodology & Insights":
    st.title("Methodology")
    st.info("LDA Topic Modeling & RoBERTa Sentiment Analysis")
    
    tab1, tab2, tab3 = st.tabs(["LDA Model", "RoBERTa Model", "EDA"])
    
    with tab1:
        st.write("LDA was selected over BERTopic for 100% data coverage.")
    with tab2:
        st.write("RoBERTa achieved 86% Accuracy in sentiment classification.")
    with tab3:
        st.write("Visualizations (EDA) would appear here.")
        # Basic Histogram
        if 'avg_rating' in df.columns:
            fig = px.histogram(df, x='avg_rating', title="Rating Distribution", color_discrete_sequence=['#355C7D'])
            st.plotly_chart(fig, use_container_width=True)
