import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
# 2. CUSTOM CSS - HIGH CONTRAST DROPDOWNS & PASTEL THEME
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
       DROPDOWN VISIBILITY FIX (White Text on Dark Background)
       -------------------------------------- */
    
    /* The Clickable Box */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background-color: #2C3E50 !important;
        color: white !important;
        border: 1px solid #5B7C99;
    }
    
    /* Text inside the clickable box */
    .stSelectbox div[data-testid="stMarkdownContainer"] p, 
    .stMultiSelect div[data-testid="stMarkdownContainer"] p {
        color: white !important;
    }
    
    /* The Dropdown Menu List */
    ul[data-baseweb="menu"] {
        background-color: #2C3E50 !important;
    }
    
    /* The Options in the list */
    li[data-baseweb="option"] {
        color: white !important;
    }
    
    /* Hover State for Options */
    li[data-baseweb="option"]:hover {
        background-color: #D4788C !important;
        color: white !important;
    }
    
    /* Selected Tags in MultiSelect */
    .stMultiSelect div[data-baseweb="tag"] {
        background-color: #D4788C !important;
    }
    .stMultiSelect div[data-baseweb="tag"] span {
        color: white !important;
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
        color: #2C3E50;
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
        background-color: #F9F9F9;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 10px;
        margin-top: 5px;
        font-style: italic;
        color: #333 !important; /* Force dark text */
        border-left: 3px solid #5B7C99;
        font-size: 0.9rem;
    }
    
    /* Topic Score Mini-Badge */
    .topic-score-badge {
        background-color: #E8EEF8;
        padding: 5px 10px;
        border-radius: 10px;
        margin-right: 5px;
        margin-bottom: 5px;
        display: inline-block;
        font-size: 0.85rem;
        color: #2C3E50;
        border: 1px solid #A5C4D4;
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
        df.columns = df.columns.str.strip()
        
        # Ratings Fix (1-5 range)
        df['avg_rating'] = pd.to_numeric(df['avg_rating'], errors='coerce').clip(1.0, 5.0)
        df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce').fillna(0).astype(int)
        
        # Topic Scores Fix
        for col in TOPIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(3.0).clip(1.0, 5.0)
        
        # Text Columns Fix
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
    st.error("⚠️ Data file 'streamlitdata.csv' not found.")
    st.stop()

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def calculate_wlc_score(row, selected_topics):
    """Calculates weighted average of selected topic scores"""
    if not selected_topics:
        return row['avg_rating']
    
    scores = []
    for topic in selected_topics:
        if topic in row:
            scores.append(row[topic])
            
    if not scores:
        return row['avg_rating']
        
    return sum(scores) / len(scores)

def get_short_reviews(row, limit=3):
    """Aggregates, deduplicates, and truncates reviews"""
    all_reviews = []
    for topic in TOPIC_COLS:
        text_col = f"{topic}_text"
        if text_col in row:
            text = str(row[text_col])
            if len(text) > 20 and "No specific mentions" not in text:
                # Split if multiple reviews concatenated
                parts = text.split('|')
                for p in parts:
                    clean_p = p.strip()
                    if len(clean_p) > 20:
                        all_reviews.append(clean_p)
    
    # Deduplicate and slice
    unique_reviews = list(set(all_reviews))
    short_reviews = []
    for r in unique_reviews[:limit]:
        # Truncate to 150 chars for "Short" view
        snippet = r[:150] + "..." if len(r) > 150 else r
        short_reviews.append(snippet)
        
    return short_reviews

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
    st.markdown("Ranked by Google Rating & Review Volume. Click card to see details.")
    st.divider()

    # Reliability Filter
    qualified_df = df[df['review_count'] >= 50].copy()
    if qualified_df.empty: qualified_df = df.copy()

    # Sort
    top_20 = qualified_df.sort_values(by=['avg_rating', 'review_count'], ascending=[False, False]).head(20)

    for i, (index, row) in enumerate(top_20.iterrows()):
        rank = i + 1
        name = row['restaurant']
        
        # Badge Logic
        badge_class = "badge-blue"
        if rank == 1: badge_class = "badge-gold"
        elif rank == 2: badge_class = "badge-silver"
        elif rank == 3: badge_class = "badge-bronze"
        
        icon = f"#{rank}" if rank > 3 else ["🥇","🥈","🥉"][rank-1]

        with st.expander(f"{icon}  {name}  —  ⭐ {row['avg_rating']:.2f}", expanded=(rank<=1)):
            
            # 1. Main Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Google Rating", f"{row['avg_rating']:.2f}")
            c2.metric("Reviews", f"{row['review_count']:,}")
            c3.markdown(f'<span class="rank-badge {badge_class}">Rank {rank}</span>', unsafe_allow_html=True)
            
            st.divider()
            
            # 2. RoBERTa Topic Breakdown
            st.markdown("**RoBERTa Aspect Ratings:**")
            topic_html = ""
            # Show top 4 strongest aspects
            sorted_topics = sorted([(t, row[t]) for t in TOPIC_COLS], key=lambda x: x[1], reverse=True)[:4]
            
            for topic, score in sorted_topics:
                topic_html += f'<span class="topic-score-badge"><b>{topic}:</b> {score:.1f}/5</span>'
            st.markdown(topic_html, unsafe_allow_html=True)
            
            # 3. Short Reviews (Max 3)
            st.markdown("**User Reviews:**")
            reviews = get_short_reviews(row, limit=3)
            if reviews:
                for rev in reviews:
                    st.markdown(f'<div class="review-box">"{rev}"</div>', unsafe_allow_html=True)
            else:
                st.caption("No text reviews available.")

# ==========================================
# PAGE 2: FIND YOUR RESTAURANT
# ==========================================
elif page == "🔍 Find Your Restaurant":
    st.title("🔍 Find Your Perfect Dining Spot")
    st.markdown("### Personalize your dining experience! 💖 Choose your preferences")
    st.divider()

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("1. Cuisine Filter")
        # Cuisine Dropdown
        cuisine_pref = st.selectbox(
            "Select Cuisine Type",
            ["All Cuisines", "Asian Cuisine", "Western Cuisine"],
            help="Filters restaurants that excel in specific cuisine types"
        )

        st.subheader("2. Priorities")
        # Multiselect for Ranking
        # We remove the cuisine types from this list to avoid redundancy, 
        # or keep them if you want users to weigh them heavily. 
        # Here we keep them in case they want to Rank by "Asian" even if filtered by "All".
        selected_priorities = st.multiselect(
            "What matters most?",
            TOPIC_COLS,
            default=["Food Quality", "Service Operations/Speed"],
            placeholder="Select aspects..."
        )
        
        st.subheader("3. Quality Filters")
        min_rating = st.slider("Min Google Rating", 1.0, 5.0, 3.5)
        
        find_btn = st.button("🚀 Find My Restaurant", use_container_width=True)

    with col2:
        if find_btn:
            # --- FILTER LOGIC ---
            filtered_df = df[df['avg_rating'] >= min_rating].copy()
            
            # Apply Cuisine Filter
            if cuisine_pref == "Asian Cuisine":
                # Must have reasonable Asian score or mention
                filtered_df = filtered_df[filtered_df['Asian Cuisine'] >= 3.0]
            elif cuisine_pref == "Western Cuisine":
                filtered_df = filtered_df[filtered_df['Western Cuisine'] >= 3.0]
            
            if filtered_df.empty:
                st.error("No restaurants found. Try relaxing the filters.")
            else:
                # --- RANKING LOGIC ---
                # Calculate WLC based on Priorities
                filtered_df['wlc_score'] = filtered_df.apply(
                    lambda row: calculate_wlc_score(row, selected_priorities), axis=1
                )
                
                # Sort by WLC
                results = filtered_df.sort_values(by='wlc_score', ascending=False).head(10)
                
                st.subheader(f"Top Recommendations ({cuisine_pref})")
                
                for idx, (i, row) in enumerate(results.iterrows()):
                    rank = idx + 1
                    wlc = row['wlc_score']
                    
                    # Card Container
                    st.markdown(f"""
                    <div class="restaurant-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span class="rank-badge badge-blue">#{rank}</span>
                                <span style="font-size:1.3rem; font-weight:bold; color:#5B7C99;">{row['restaurant']}</span>
                            </div>
                            <div style="text-align:right;">
                                <span style="font-size:1.6rem; font-weight:bold; color:#D4788C;">{wlc:.1f}</span>
                                <span style="font-size:0.8rem; color:#888;">Match Score</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Aspect Breakdown Grid
                    if selected_priorities:
                        st.markdown("<div style='margin-top:10px; margin-bottom:10px;'>", unsafe_allow_html=True)
                        cols = st.columns(len(selected_priorities))
                        for c_idx, topic in enumerate(selected_priorities):
                            score = row.get(topic, 0)
                            with cols[c_idx]:
                                st.caption(topic)
                                st.markdown(f"**{score:.1f}**")
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PAGE 3: METHODOLOGY
# ==========================================
elif page == "📊 Methodology & Insights":
    st.title("📊 Methodology & Insights")
    
    tab1, tab2, tab3 = st.tabs(["LDA vs BERTopic", "RoBERTa Analysis", "EDA"])
    
    # TAB 1: MODEL COMPARISON
    with tab1:
        st.subheader("Why LDA?")
        st.markdown("We compared **Latent Dirichlet Allocation (LDA)** against **BERTopic**. LDA was selected for the final application due to superior data coverage.")
        
        # Comparison Table
        comp_data = {
            'Metric': ['Data Coverage', 'Topic Volume', 'Outliers'],
            'LDA': ['100%', '7 (Optimized)', '0%'],
            'BERTopic': ['56.1%', '81 (Noisy)', '43.9%']
        }
        df_comp = pd.DataFrame(comp_data)
        
        # Styled Table
        st.table(df_comp)
        
        st.info("💡 **Conclusion:** BERTopic discarded nearly half the dataset as 'outliers', making it unsuitable for a recommendation system where every restaurant needs to be scored.")

    # TAB 2: ROBERTA
    with tab2:
        st.subheader("Sentiment Quantifiction")
        st.markdown("We used `cardiffnlp/twitter-roberta-base-sentiment` to score each topic.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Performance Metrics")
            st.metric("Accuracy", "87.03%")
            st.metric("F1-Score", "85.00%")
        
        with col2:
            st.markdown("#### Class Performance")
            metrics_df = pd.DataFrame({
                "Class": ["Positive", "Negative"],
                "Precision": ["93%", "62%"],
                "Recall": ["97%", "84%"]
            })
            st.dataframe(metrics_df, hide_index=True)

    # TAB 3: EDA
    with tab3:
        st.subheader("Exploratory Analysis")
        c1, c2 = st.columns(2)
        with c1: 
            st.markdown("**Word Cloud**")
            # Try multiple paths
            if os.path.exists("wordcloud.png"): st.image("wordcloud.png")
            elif os.path.exists("images/wordcloud.png"): st.image("images/wordcloud.png")
            else: st.warning("Image not found")
            
        with c2:
            st.markdown("**N-Gram**")
            if os.path.exists("ngram.png"): st.image("ngram.png")
            elif os.path.exists("images/ngram.png"): st.image("images/ngram.png")
            else: st.warning("Image not found")

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>KL Dining Assistant © 2025</div>", unsafe_allow_html=True)
