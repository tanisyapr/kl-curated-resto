import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
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
# 2. CUSTOM CSS - PASTEL PINK & BLUE THEME
# ==========================================
st.markdown("""
<style>
    /* Main background - soft pastel gradient */
    .stApp { 
        background: linear-gradient(135deg, #E8F4F8 0%, #F8E8F0 50%, #E8EEF8 100%); 
        color: #2C3E50; 
    }
    
    /* Typography */
    h1 { color: #5B7C99 !important; font-weight: 800 !important; }
    h2, h3 { color: #7C5B7D !important; font-weight: 700 !important; }
    p, label, span, div { color: #2C3E50; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #5B7C99 0%, #7C5B7D 100%);
    }
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
    
    /* --------------------------------------
       DROPDOWN & INPUT VISIBILITY FIXES 
       -------------------------------------- */
    /* Input boxes */
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 2px solid #5B7C99 !important;
        color: #2C3E50 !important;
    }
    
    /* The selected tags inside the box */
    .stMultiSelect div[data-baseweb="tag"] {
        background-color: #D4788C !important;
    }
    .stMultiSelect div[data-baseweb="tag"] span {
        color: #FFFFFF !important;
        font-weight: bold !important;
    }
    
    /* The dropdown list items */
    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }
    ul[data-baseweb="menu"] li {
        color: #2C3E50 !important; /* Dark text for contrast */
        background-color: #FFFFFF !important;
    }
    ul[data-baseweb="menu"] li:hover {
        background-color: #F8E8F0 !important; /* Pink hover */
        font-weight: bold !important;
    }
    
    /* --------------------------------------
       RESTAURANT CARDS 
       -------------------------------------- */
    .restaurant-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 6px solid #D4788C;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .restaurant-card:hover {
        transform: scale(1.01);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    /* Rank Badges */
    .rank-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin-right: 10px;
        font-size: 0.9rem;
    }
    .badge-gold { background: linear-gradient(135deg, #FFD700, #FFA500); }
    .badge-silver { background: linear-gradient(135deg, #C0C0C0, #A9A9A9); }
    .badge-bronze { background: linear-gradient(135deg, #CD7F32, #8B4513); }
    .badge-blue { background: linear-gradient(135deg, #5B7C99, #A5C4D4); }

    /* Review Box */
    .review-box {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        font-style: italic;
        color: #555;
        border-left: 3px solid #5B7C99;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #D4788C !important;
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #D4788C 0%, #7C5B7D 100%);
        color: white !important;
        border: none;
        border-radius: 25px;
        font-weight: bold;
        padding: 0.5rem 2rem;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA LOADING & CLEANING
# ==========================================
# Define your exact topics based on your previous processing
TOPIC_COLS = [
    'Ambiance & Atmosphere',
    'Staff Friendliness',
    'Asian Cuisine',
    'Management',
    'Service Operations/Speed',
    'Western Cuisine',
    'Food Quality'
]

@st.cache_data
def load_and_clean_data():
    filename = 'streamlitdata.csv'
    if not os.path.exists(filename):
        return None
    
    try:
        df = pd.read_csv(filename)
        
        # 1. Clean Column Names
        df.columns = df.columns.str.strip()
        
        # 2. Fix Ratings (The 12/5 Bug Fix)
        # Force convert to numeric, coerce errors to NaN
        df['avg_rating'] = pd.to_numeric(df['avg_rating'], errors='coerce')
        # Clip values strictly between 1.0 and 5.0
        df['avg_rating'] = df['avg_rating'].clip(1.0, 5.0)
        
        # 3. Fix Review Counts
        df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce').fillna(0).astype(int)
        
        # 4. Fix Topic Scores
        for col in TOPIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(3.0).clip(1.0, 5.0)
                
        # 5. Ensure Text Columns Exist (Avoid KeyErrors)
        for col in TOPIC_COLS:
            text_col = f"{col}_text"
            if text_col not in df.columns:
                df[text_col] = "No specific mentions."
            else:
                df[text_col] = df[text_col].fillna("No specific mentions.")
                
        return df
        
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None

df = load_and_clean_data()

if df is None:
    st.error("⚠️ Data file 'streamlitdata.csv' not found. Please upload the file generated from the previous step.")
    st.stop()

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def calculate_wlc_score(row, selected_topics):
    """Weighted Linear Combination Score (1-5)"""
    if not selected_topics:
        return row['avg_rating']
    
    scores = []
    for topic in selected_topics:
        if topic in row:
            scores.append(row[topic])
            
    if not scores:
        return row['avg_rating']
        
    return sum(scores) / len(scores)

def get_aggregated_reviews(row, selected_topics=None):
    """Collects reviews from the text columns of selected topics"""
    reviews = []
    
    # If specific topics selected, look there first
    targets = selected_topics if selected_topics else TOPIC_COLS
    
    for topic in targets:
        text_col = f"{topic}_text"
        if text_col in row:
            text = str(row[text_col])
            # Filter out placeholders
            if len(text) > 15 and "No specific mentions" not in text:
                reviews.append(text)
                
    # Deduplicate and limit
    return list(set(reviews))[:3]

# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("# 🍽️ KL Dining")
st.sidebar.markdown("### Smart Recommendations")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    ["🏆 Best of The Best", "🔍 Find Your Restaurant", "📊 Methodology & Insights"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Thesis Project** Master of Data Science  
    University of Malaya  
    **Tanisya Pristi Azrelia** (24088031)
    """
)

# ==========================================
# PAGE 1: BEST OF THE BEST
# ==========================================
if page == "🏆 Best of The Best":
    st.title("🏆 Best of The Best")
    st.markdown("### Top 20 Highest-Rated Restaurants in Kuala Lumpur")
    st.markdown("These rankings are based on overall Google ratings and review volume. **Click on a restaurant card to read what people are saying.**")
    st.divider()

    # Filter for reliability (e.g., at least 50 reviews)
    qualified_df = df[df['review_count'] >= 50].copy()
    if qualified_df.empty:
        qualified_df = df.copy() # Fallback

    # Sort by Rating desc, then Review Count desc
    top_20 = qualified_df.sort_values(by=['avg_rating', 'review_count'], ascending=[False, False]).head(20)

    for i, (index, row) in enumerate(top_20.iterrows()):
        rank = i + 1
        name = row['restaurant']
        rating = row['avg_rating']
        count = row['review_count']
        
        # Badge Logic
        if rank == 1:
            badge_class = "badge-gold"
            icon = "🥇"
        elif rank == 2:
            badge_class = "badge-silver"
            icon = "🥈"
        elif rank == 3:
            badge_class = "badge-bronze"
            icon = "🥉"
        else:
            badge_class = "badge-blue"
            icon = f"#{rank}"

        # Card Content (Clickable Expander)
        with st.expander(f"{icon}  {name}  —  ⭐ {rating:.1f}/5.0", expanded=(rank<=3)):
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Google Rating", f"{rating:.2f}")
            c2.metric("Total Reviews", f"{count:,}")
            c3.markdown(f"""
                <div style="text-align:center; padding-top:10px;">
                    <span class="rank-badge {badge_class}" style="font-size:1.2rem;">Rank {rank}</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 💬 What people say:")
            
            # Get reviews from all topics to show a mix
            reviews = get_aggregated_reviews(row)
            
            if reviews:
                for rev in reviews:
                    st.markdown(f'<div class="review-box">"{rev}"</div>', unsafe_allow_html=True)
            else:
                st.info("No detailed text reviews available in the analysis dataset.")

# ==========================================
# PAGE 2: FIND YOUR RESTAURANT
# ==========================================
elif page == "🔍 Find Your Restaurant":
    st.title("🔍 Find Your Perfect Dining Spot")
    st.markdown("### Customized Recommendations using AI")
    st.markdown("Select what matters most to you, and our WLC (Weighted Linear Combination) algorithm will rank the best matches.")
    st.divider()

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("1. Your Preferences")
        
        # MULTISELECT (Contrast fixed in CSS)
        selected_priorities = st.multiselect(
            "What are you looking for?",
            options=TOPIC_COLS,
            default=None,
            placeholder="Select aspects (e.g. Food Quality)"
        )
        
        st.subheader("2. Filters")
        min_rating_filter = st.slider("Minimum Google Rating", 1.0, 5.0, 3.5, 0.1)
        min_reviews_filter = st.slider("Minimum Review Count", 10, 500, 50, 10)
        
        find_btn = st.button("🚀 Find My Restaurant", use_container_width=True)

    with col2:
        if find_btn:
            if not selected_priorities:
                st.warning("⚠️ Please select at least one preference in the sidebar to get started!")
            else:
                st.subheader(f"Top 10 Recommendations")
                st.caption(f"Based on: {', '.join(selected_priorities)}")
                
                # 1. Apply Filters
                filtered = df[
                    (df['avg_rating'] >= min_rating_filter) & 
                    (df['review_count'] >= min_reviews_filter)
                ].copy()
                
                if filtered.empty:
                    st.error("No restaurants found matching your filters. Try lowering the rating or review count.")
                else:
                    # 2. Calculate Match Score
                    filtered['match_score'] = filtered.apply(
                        lambda row: calculate_wlc_score(row, selected_priorities), axis=1
                    )
                    
                    # 3. Sort
                    results = filtered.sort_values(by=['match_score', 'avg_rating'], ascending=[False, False]).head(10)
                    
                    # 4. Display Cards
                    for idx, (i, row) in enumerate(results.iterrows()):
                        rank = idx + 1
                        score = row['match_score']
                        
                        st.markdown(f"""
                        <div class="restaurant-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span class="rank-badge badge-blue">#{rank}</span>
                                    <span style="font-size:1.3rem; font-weight:bold; color:#5B7C99;">{row['restaurant']}</span>
                                </div>
                                <div style="text-align:right;">
                                    <span style="font-size:1.5rem; font-weight:bold; color:#D4788C;">{score:.1f}</span>
                                    <span style="font-size:0.8rem; color:#888;">/ 5.0 Match</span>
                                </div>
                            </div>
                            <hr style="margin:10px 0;">
                            <div style="display:flex; gap:15px; margin-bottom:10px;">
                                <span>⭐ <b>{row['avg_rating']:.1f}</b> Google Rating</span>
                                <span>📝 <b>{row['review_count']}</b> Reviews</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Show relevant snippet
                        relevant_reviews = get_aggregated_reviews(row, selected_priorities)
                        if relevant_reviews:
                            snippet = relevant_reviews[0]
                            # Bold the selected keywords if possible (simple find/replace)
                            st.markdown(f'<div class="review-box" style="font-size:0.9rem;">"{snippet}"</div>', unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PAGE 3: METHODOLOGY
# ==========================================
elif page == "📊 Methodology & Insights":
    st.title("📊 Methodology & Insights")
    st.markdown("### How this system works")
    
    tabs = st.tabs(["🧠 Model Metrics", "🔍 EDA (WordCloud)"])
    
    with tabs[0]:
        st.subheader("RoBERTa Sentiment Analysis Performance")
        st.markdown("We employed `cardiffnlp/twitter-roberta-base-sentiment` to analyze review sentiment. Below are the evaluation metrics against ground truth labels.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", "87.03%")
            st.metric("F1-Score (Weighted)", "85.00%")
        
        with col2:
            # Recreating the metrics table from your data
            metrics_df = pd.DataFrame({
                "Metric": ["Precision (Positive)", "Precision (Negative)", "Recall (Positive)", "Recall (Negative)"],
                "Score": ["93%", "62%", "97%", "84%"]
            })
            st.table(metrics_df)

        st.markdown("### Topic Modeling (LDA)")
        st.info("LDA (Latent Dirichlet Allocation) was selected over BERTopic for this application because it achieved **100% data coverage**, whereas BERTopic classified 45% of reviews as outliers.")

    with tabs[1]:
        st.subheader("Exploratory Data Analysis")
        
        c1, c2 = st.columns(2)
        
        # You need to make sure these image files exist in your folder
        with c1:
            st.markdown("**Word Cloud**")
            if os.path.exists("wordcloud.png"):
                st.image("wordcloud.png", use_container_width=True)
            elif os.path.exists("images/wordcloud.png"):
                st.image("images/wordcloud.png", use_container_width=True)
            else:
                st.warning("image 'wordcloud.png' not found.")
                
        with c2:
            st.markdown("**N-Gram Analysis**")
            if os.path.exists("ngram.png"):
                st.image("ngram.png", use_container_width=True)
            elif os.path.exists("images/ngram.png"):
                st.image("images/ngram.png", use_container_width=True)
            else:
                st.warning("image 'ngram.png' not found.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>KL Dining Assistant © 2025</div>", unsafe_allow_html=True)
