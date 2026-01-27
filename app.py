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
# 3. DATA LOADING & CLEANING
# ==========================================
@st.cache_data
def load_data():
    filename = 'streamlitdata.csv'
    if not os.path.exists(filename):
        return None
    try:
        df = pd.read_csv(filename)
        # 1. Strip whitespace from column names (prevents "Topic " vs "Topic" errors)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        return None

df = load_data()

if df is None:
    st.error("CRITICAL ERROR: 'streamlitdata.csv' not found. Please upload it.")
    st.stop()

# ==========================================
# 4. DYNAMIC TOPIC DETECTION (THE FIX)
# ==========================================
# We define what columns are NOT topics. Everything else is treated as a topic.
non_topic_cols = [
    'restaurant', 'avg_rating', 'review_count', 'review', 
    'topic', 'topic_label', 'topic_id', 'lda_topic_id', 'final_score', 
    'Western Cuisine', 'Asian Cuisine' # Exclude cuisine filters from the preference list
]

# Automatically find numeric columns that are likely your topics
available_topics = [
    col for col in df.columns 
    if col not in non_topic_cols 
    and pd.api.types.is_numeric_dtype(df[col])
]

# ==========================================
# 5. SIDEBAR
# ==========================================
st.sidebar.title("KL Dining Assistant")
st.sidebar.markdown("By Tanisya Pristi Azrelia")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Best of The Best", "Find Your Restaurant", "Methodology & Insights"])

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "Best of The Best":
    st.title("KL Restaurant Recommendation System")
    st.markdown("Welcome! Explore the top-rated establishments below.")
    st.divider()
    
    if 'review_count' in df.columns and 'avg_rating' in df.columns:
        top_restaurants = df[df['review_count'] > 50].sort_values('avg_rating', ascending=False).head(20)
        st.dataframe(
            top_restaurants,
            column_order=["restaurant", "avg_rating", "review_count"],
            hide_index=True,
            use_container_width=True,
            height=600
        )

# ==========================================
# PAGE 2: RECOMMENDATION ENGINE
# ==========================================
elif page == "Find Your Restaurant":
    st.title("Personalized Recommendation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 1. Select Preferences")
        
        # ERROR CHECK: If no topics found, warn the user
        if not available_topics:
            st.error("No topic columns found! Check your CSV file.")
            st.write("Columns detected:", list(df.columns))
            st.stop()

        # DYNAMIC DROPDOWN: Only shows what exists in 'available_topics'
        priorities = st.multiselect(
            "What matters most?",
            options=available_topics,
            default=[available_topics[0]] if available_topics else None
        )
        
        st.markdown("### 2. Cuisine Filter")
        cuisine_pref = st.radio("Category:", ["All Cuisines", "Western", "Asian"])
        
        st.markdown("---")
        btn = st.button("Find Best Place", type="primary")

    with col2:
        if btn:
            # 1. SCORING
            if priorities:
                df['final_score'] = df[priorities].mean(axis=1)
            else:
                df['final_score'] = df.get('avg_rating', 0)
            
            # 2. FILTERING
            filtered_df = df.copy()
            
            # Robust Cuisine Filtering
            if cuisine_pref == "Western":
                # Check if specific column exists, otherwise skip filter
                west_col = next((c for c in df.columns if "Western" in c), None)
                if west_col:
                    filtered_df = filtered_df[filtered_df[west_col] > 3.0]
                    
            elif cuisine_pref == "Asian":
                asian_col = next((c for c in df.columns if "Asian" in c), None)
                if asian_col:
                    filtered_df = filtered_df[filtered_df[asian_col] > 3.0]

            # 3. DISPLAY
            results = filtered_df.sort_values('final_score', ascending=False).head(5)
            
            if priorities:
                st.subheader(f"Top Recommendations for: {', '.join(priorities)}")
            else:
                st.subheader("Top Recommendations")

            if len(results) == 0:
                st.warning("No matches found.")
                
            for i, (index, row) in enumerate(results.iterrows()):
                with st.container():
                    st.markdown(f"### #{i+1} {row['restaurant']}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Match Score", f"{row['final_score']:.2f}")
                    
                    if 'avg_rating' in row:
                        c2.metric("Rating", f"{row['avg_rating']:.1f}")
                    
                    if 'review_count' in row:
                        c3.metric("Reviews", f"{int(row['review_count'])}")
                        
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
        st.write("LDA selected for 100% data coverage.")
    with tab2:
        st.write("RoBERTa accuracy: 86.31%.")
    with tab3:
        if 'avg_rating' in df.columns:
            fig = px.histogram(df, x='avg_rating', title="Rating Distribution", color_discrete_sequence=['#355C7D'])
            st.plotly_chart(fig, use_container_width=True)
