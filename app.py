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

    /* 4. INFO/SUCCESS BOXES */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.95);
        color: #355C7D;
        border-radius: 10px;
        border: 2px solid #F8B195;
    }

    /* 5. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #2A3E50;
        color: #FFFFFF;
    }
    
    /* 6. BUTTONS */
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

    /* 7. WIDGETS */
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
        background-color: #F8B195;
    }
    span[data-baseweb="tag"] {
        background-color: #F8B195 !important;
        color: #355C7D !important;
    }
    
    /* 8. TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
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
    filename = 'streamlitdata_with_text.csv'
    if not os.path.exists(filename):
        filename = 'streamlitdata.csv'
        
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("Data file not found! Please upload 'streamlitdata_with_text.csv' to your repository.")
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
    **Welcome!** This platform uses advanced machine learning (LDA & RoBERTa) to analyze thousands of reviews.
    Your dining experience should be personal, so get your recommendation here by choosing your preferences.
    """)
    
    st.divider()
    st.subheader("The Hall of Fame (Top 20)")
    st.info("These restaurants have Ratings > 4.0 and High Reliability (>50 reviews).")

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
    st.title("Where should you eat today in Kuala Lumpur?")
    st.markdown("Choose your dining priorities below")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 1. Select Your Priorities")
        
        # MULTI-SELECT
        priorities = st.multiselect(
            "What matters most to you?",
            ["Food Quality", "Value for Money", "Staff Friendliness", "Service Speed", "Dining Experience"],
            default=["Food Quality"]
        )
        
        # Convert selection to weights
        w_food = 1.0 if "Food Quality" in priorities else 0.0
        w_value = 1.0 if "Value for Money" in priorities else 0.0
        w_staff = 1.0 if "Staff Friendliness" in priorities else 0.0
        w_speed = 1.0 if "Service Speed" in priorities else 0.0
        w_exp = 1.0 if "Dining Experience" in priorities else 0.0
        
        st.markdown("### 2. Cuisine Filter")
        cuisine_pref = st.radio("Select Type:", ["All Cuisines", "Western/Italian", "Asian/Local"])
        
        st.markdown("---")
        btn = st.button("Find My Match", type="primary")

    with col2:
        if btn:
            # --- SMART COLUMN FINDER ---
            def get_val(row, target_name):
                # List of possible column names in your CSV
                possible_names = [
                    target_name, 
                    target_name.lower(), 
                    target_name.replace(" ", "_").lower(), # e.g. "food_quality"
                    target_name.replace(" ", "_")          # e.g. "Food_Quality"
                ]
                # Return the first one found
                for name in possible_names:
                    if name in row:
                        return row[name]
                return 0.0 # Return 0 if not found

            # 1. CALCULATE SCORE
            def calculate_score(row):
                s = 0.0
                s += get_val(row, 'Food Quality') * w_food
                s += get_val(row, 'Value for Money') * w_value
                s += get_val(row, 'Staff Friendliness') * w_staff
                s += get_val(row, 'Service Speed') * w_speed
                s += get_val(row, 'Dining Experience') * w_exp
                return s

            df['final_score'] = df.apply(calculate_score, axis=1)
            
            # 2. FILTER BY CUISINE
            filtered_df = df.copy()
            
            if cuisine_pref == "Western/Italian":
                west_col = next((c for c in ['Western Cuisine', 'western_cuisine', 'Western_Cuisine'] if c in df.columns), None)
                if west_col:
                    filtered_df = filtered_df[filtered_df[west_col] > 3.0]
                    
            elif cuisine_pref == "Asian/Local":
                asian_col = next((c for c in ['Asian Cuisine', 'asian_cuisine', 'Asian_Cuisine'] if c in df.columns), None)
                if asian_col:
                    filtered_df = filtered_df[filtered_df[asian_col] > 3.0]
                
            # 3. SORT & DISPLAY TOP 5
            results = filtered_df.sort_values('final_score', ascending=False).head(5)
            
            st.subheader("Top 5 Recommendations")
            
            if len(results) == 0:
                st.warning("No matches found. Try selecting 'All Cuisines' or adding priorities!")
                
            for i, (index, row) in enumerate(results.iterrows()):
                # Create a card-like container
                with st.container():
                    st.markdown(f"### #{i+1} {row['restaurant']}")
                    
                    # Metrics Row
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Match Score", f"{row['final_score']:.2f}")
                    c2.metric("Stars", f"{row['avg_rating']:.1f}")
                    
                    # Show Food Score if available
                    food_val = f"{get_val(row, 'Food Quality'):.1f}"
                    c3.metric("Food Score", food_val)
                    
                    # Show Value Score if available
                    val_val = f"{get_val(row, 'Value for Money'):.1f}"
                    c4.metric("Value Score", val_val)
                    
                    # Progress Bar
                    max_possible = (w_food + w_value + w_staff + w_speed + w_exp) * 5
                    if max_possible > 0:
                        norm_score = row['final_score'] / max_possible
                        norm_score = min(1.0, max(0.0, norm_score)) # Clamp
                        st.progress(norm_score)
                    
                    # Review Evidence (NO EMOJIS)
                    with st.expander("See what people actually said (Evidence)"):
                        if w_food > 0 and 'Food Quality_text' in row:
                            st.markdown(f"**Food:** _{row['Food Quality_text']}_")
                        if w_staff > 0 and 'Staff Friendliness_text' in row:
                            st.markdown(f"**Staff:** _{row['Staff Friendliness_text']}_")
                        if w_value > 0 and 'Value for Money_text' in row:
                            st.markdown(f"**Value:** _{row['Value for Money_text']}_")
                        if w_speed > 0 and 'Service Speed_text' in row:
                            st.markdown(f"**Speed:** _{row['Service Speed_text']}_")
                        if w_exp > 0 and 'Dining Experience_text' in row:
                            st.markdown(f"**Vibe:** _{row['Dining Experience_text']}_")

                    st.markdown("---")

# ==========================================
# PAGE 3: METHODOLOGY & INSIGHTS
# ==========================================
elif page == "Methodology & Insights":
    st.title("Methodology & Analysis")
    st.markdown("This is the methodology used for this analysis.")
    
    tab1, tab2, tab3 = st.tabs(["Topic Modeling (LDA)", "Sentiment (RoBERTa)", "Exploratory Analysis (EDA)"])
    
    # --- TAB 1: LDA vs BERTopic ---
    with tab1:
        st.header("Why LDA?")
        st.markdown("We compared LDA vs BERTopic. **LDA was chosen** for its ability to generalize better for broad categories like 'Service' and 'Value'.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.success("LDA (Selected)")
            st.markdown("- 100% Coverage\n- Stable Topics\n- Better Interpretability")
        with c2:
            st.error("BERTopic (Discarded)")
            st.markdown("- 45% Data Loss (Outliers)\n- Generate 100+ topics\n- LDA will be harder to deploy and confusing to the users to choose that many topics")

    # --- TAB 2: RoBERTa ---
    with tab2:
        st.header("RoBERTa Sentiment Analysis")
        st.markdown("We fine-tuned `twitter-roberta-base-sentiment` to score reviews on a 1-5 scale.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Accuracy", "86.31%")
        col2.metric("Precision (Positive)", "93%")
        col3.metric("Recall (Positive)", "96%")
        
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Confusion Matrix", width=500)
        else:
            st.info("Confusion Matrix image not uploaded.")

    # --- TAB 3: EDA ---
    with tab3:
        st.header("Exploratory Data Analysis")
        st.markdown("Visualizing the dataset to understand underlying patterns in customer feedback.")
        
        # ROW 1: Rating Distribution & Correlation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Rating Distribution")
            # Live Chart Generation
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
            **Insight:** The dataset is left-skewed, meaning most restaurants in KL have positive ratings (> 4.0). 
            This highlights the need for **Sentiment Analysis** to distinguish "Good" from "Great".
            """)

        with col2:
            st.subheader("2. Correlation Analysis")
            # Select numeric columns for correlation
            corr_cols = ['avg_rating', 'food_quality', 'service', 'value', 'western_cuisine', 'asian_cuisine']
            
            # Check if columns exist
            available_cols = [c for c in corr_cols if c in df.columns]
            
            if len(available_cols) > 1:
                # Compute correlation matrix live
                corr_matrix = df[available_cols].corr()
                
                # Heatmap
                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Correlation: Sentiment vs. Overall Rating"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            st.info("""
            **Insight:** **Food Quality** and **Service** typically show the strongest positive correlation with the Overall Rating. 
            Value for money often has a weaker impact on the final star rating compared to taste and service.
            """)

        st.markdown("---")

        # ROW 2: Wordcloud & N-Grams
        st.subheader("3. Text Analysis (Word Cloud & N-Grams)")
        st.markdown("Most frequent words and phrases used by customers.")

        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Word Cloud**")
            import os
            if os.path.exists("wordcloud.png"):
                st.image("wordcloud.png", caption="Most Common Words in Reviews", use_container_width=True)
                st.markdown("""
                **Interpretation:** Dominant terms like *"Delicious", "Service",* and *"Friendly"* confirm that the dataset is heavily focused on dining experiences rather than logistics.
                """)
            else:
                st.warning("⚠️ 'wordcloud.png' not found. Please upload the image from your EDA Analysis.")

        with c2:
            st.markdown("**Top Bigrams (2-Words Phrases) and Trigrams (3-Words Phrases)**")
            if os.path.exists("ngram.png"):
                st.image("ngram.png", caption="Top Bigrams (2-Words Phrases) and Trigrams (3-Words Phrases)", use_container_width=True)
                st.markdown("""
                **Interpretation:** Phrases like *"Nasi Lemak"* and *"Soft Shell Crab"* identify popular local dishes, while *"Friendly Staff"* and *"Staff Friendly Helpful" appears frequently, validating the LDA topic modeling results.
                """)
            else:
                st.warning("⚠️ 'ngram.png' not found. Please upload the image from your EDA Analysis.")
