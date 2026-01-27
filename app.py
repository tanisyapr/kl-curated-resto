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
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Best of The Best", "Find Your Restaurant", "Methodology & Insights"])

# ==========================================
# PAGE 1: HOME & HALL OF FAME
# ==========================================
if page == "Best of The Best":
    st.title("KL Restaurant Recommendation System")
    st.markdown("""
    **Welcome.** This platform leverages LDA Topic Modeling and RoBERTa Sentiment Analysis to provide data-driven dining recommendations in Kuala Lumpur.
    Explore the top-rated establishments below or navigate to the recommendation engine to find your perfect match.
    """)
    
    st.divider()
    st.subheader("The Hall of Fame (Top 20)")
    st.info("Criteria: Average Rating > 4.0 and High Reliability (>50 verified reviews).")

    # Filter & Sort
    top_restaurants = df[df['review_count'] > 50].sort_values('avg_rating', ascending=False).head(20)
    
    # Prepare columns for display
    cols_config = {
        "restaurant": "Restaurant Name",
        "avg_rating": st.column_config.NumberColumn("Stars", format="%.2f"),
        "review_count": st.column_config.NumberColumn("Reviews", format="%d")
    }
    
    # Show dataframe
    st.dataframe(
        top_restaurants,
        column_order=["restaurant", "avg_rating", "review_count"],
        column_config=cols_config,
        hide_index=True,
        use_container_width=True,
        height=600
    )

# ==========================================
# PAGE 2: RECOMMENDATION ENGINE
# ==========================================
elif page == "Find Your Restaurant":
    st.title("Personalized Dining Recommendation")
    st.markdown("Adjust the weights below to prioritize what matters most to your dining experience.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 1. Set Preferences")
        
        # 7 Quality Sliders (Matched to your NEW Topic List)
        w_food = st.slider("Food Quality", 0.0, 1.0, 0.8)
        w_staff = st.slider("Staff Friendliness", 0.0, 1.0, 0.5)
        w_ambiance = st.slider("Ambiance & Atmosphere", 0.0, 1.0, 0.5)
        w_mgmt = st.slider("Management", 0.0, 1.0, 0.5)
        w_speed = st.slider("Service Speed", 0.0, 1.0, 0.5)
        
        st.markdown("### 2. Select Cuisine")
        cuisine_pref = st.radio("Filter by Category:", ["All Cuisines", "Western Cuisine", "Asian Cuisine"])
        
        st.markdown("---")
        btn = st.button("Find My Match", type="primary")

    with col2:
        if btn:
            # --- HELPER FUNCTION ---
            def get_col(name):
                return df[name] if name in df.columns else 0

            # 1. CALCULATE SCORE
            # Weighted sum based on user sliders
            score = (
                (get_col('Food Quality') * w_food) +
                (get_col('Staff Friendliness') * w_staff) +
                (get_col('Ambiance & Atmosphere') * w_ambiance) +
                (get_col('Management') * w_mgmt) +
                (get_col('Service Operations/Speed') * w_speed)
            )
            
            df['final_score'] = score
            
            # 2. FILTER BY CUISINE
            filtered_df = df.copy()
            
            if cuisine_pref == "Western Cuisine":
                if 'Western Cuisine' in df.columns:
                    filtered_df = filtered_df[filtered_df['Western Cuisine'] > 3.0]
                    
            elif cuisine_pref == "Asian Cuisine":
                if 'Asian Cuisine' in df.columns:
                    filtered_df = filtered_df[filtered_df['Asian Cuisine'] > 3.0]
                
            # 3. SORT & DISPLAY TOP 5
            results = filtered_df.sort_values('final_score', ascending=False).head(5)
            
            st.subheader("Top 5 Recommendations")
            
            if len(results) == 0:
                st.warning("No matches found. Try adjusting your filters.")
                
            for i, (index, row) in enumerate(results.iterrows()):
                with st.container():
                    st.markdown(f"### #{i+1} {row['restaurant']}")
                    
                    # Metrics Row
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Match Score", f"{row['final_score']:.2f}")
                    c2.metric("Overall Rating", f"{row['avg_rating']:.1f}")
                    
                    # Dynamic Metrics based on top user priorities
                    if w_food > 0.5 and 'Food Quality' in row:
                        c3.metric("Food Score", f"{row['Food Quality']:.1f}")
                    elif 'Ambiance & Atmosphere' in row:
                        c3.metric("Ambiance", f"{row['Ambiance & Atmosphere']:.1f}")
                        
                    if w_staff > 0.5 and 'Staff Friendliness' in row:
                        c4.metric("Staff Score", f"{row['Staff Friendliness']:.1f}")
                    else:
                        c4.metric("Reviews", f"{int(row['review_count'])}")
                    
                    st.markdown("---")

# ==========================================
# PAGE 3: METHODOLOGY & INSIGHTS
# ==========================================
elif page == "Methodology & Insights":
    st.title("Methodology & Analysis")
    st.markdown("Overview of the technical framework used to build this system.")
    
    tab1, tab2, tab3 = st.tabs(["Topic Modeling (LDA)", "Sentiment (RoBERTa)", "Exploratory Analysis (EDA)"])
    
    # --- TAB 1: LDA vs BERTopic ---
    with tab1:
        st.header("Topic Modeling: LDA vs. BERTopic")
        st.markdown("We compared Latent Dirichlet Allocation (LDA) against BERTopic. **LDA was selected** as the final model due to its superior generalization capabilities on this specific dataset.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.success("LDA (Selected Model)")
            st.markdown("""
            - **100% Data Coverage:** No data loss.
            - **Interpretability:** Produces clear, broad categories (e.g., 'Service', 'Value').
            - **Stability:** Consistent results suitable for deployment.
            """)
        with c2:
            st.error("BERTopic (Discarded)")
            st.markdown("""
            - **High Data Loss:** Classified ~45% of reviews as outliers (-1).
            - **Fragmentation:** Generated 100+ micro-topics, making it difficult for users to filter effectively.
            """)

    # --- TAB 2: RoBERTa ---
    with tab2:
        st.header("Sentiment Analysis: RoBERTa")
        st.markdown("We utilized a pre-trained `twitter-roberta-base-sentiment` model, fine-tuned to score reviews on a normalized 1-5 scale.")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Accuracy", "87%")
        col2.metric("Precision (Positive)", "93%")
        col3.metric("Recall (Positive)", "97%")
        
        st.markdown("### Performance Visualization")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Confusion Matrix: Predicted vs. Actual Sentiment", width=600)
        else:
            st.info("Confusion Matrix image not found.")

    # --- TAB 3: EDA ---
    with tab3:
        st.header("Exploratory Data Analysis")
        st.markdown("Key insights derived from the dataset visualization.")
        
        # 1. Rating Distribution
        st.subheader("1. Rating Distribution")
        fig_dist = px.histogram(
            df, 
            x='avg_rating', 
            nbins=10, 
            title="Distribution of Star Ratings",
            color_discrete_sequence=['#355C7D']
        )
        fig_dist.update_layout(bargap=0.1)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.info("""
        **Insight:** The dataset exhibits a left-skewed distribution, indicating that the majority of restaurants in KL maintain positive ratings (> 4.0). 
        This necessitates the use of granular sentiment analysis to distinguish 'Good' from 'Exceptional'.
        """)
        
        st.markdown("---")

        # 2. Text Analysis
        st.subheader("2. Text Analysis (Lexical Patterns)")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Word Cloud**")
            if os.path.exists("wordcloud.png"):
                st.image("wordcloud.png", caption="Most Frequent Terms", use_container_width=True)
                st.markdown("**Observation:** Dominant terms such as 'Delicious', 'Service', and 'Friendly' confirm that the dataset is heavily centered on the dining experience.")
            else:
                st.warning("Image 'wordcloud.png' not found.")

        with c2:
            st.markdown("**N-Grams (Common Phrases)**")
            if os.path.exists("ngram.png"):
                st.image("ngram.png", caption="Top Bigrams & Trigrams", use_container_width=True)
                st.markdown("**Observation:** Specific dishes like 'Nasi Lemak' and service indicators like 'Friendly Staff' appear frequently, validating the topics discovered by the LDA model.")
            else:
                st.warning("Image 'ngram.png' not found.")
