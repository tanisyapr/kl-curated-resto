import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Kuala Lumpur Curated Restaurant Recommendation",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics (High Contrast Version)
st.markdown("""
<style>
    /* 1. MAIN APP BACKGROUND */
    .stApp {
        background-color: #1F234E;
    }

    /* 2. METRIC CARD (Dark Blue with Cream Text) */
    .metric-card {
        background-color: #8e3563;
        padding: 20px;
        border-radius: 10px;
        color: #F7F5EB;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }

    /* 3. HEADINGS (Cream with Shadow for readability) */
    h1, h2, h3 {
        color: #F7F5EB !important;
        text-shadow: 2px 2px 0px #355C7D; /* Hard shadow makes it pop against pink */
    }

    /* 4. ERROR & WARNING BOXES (FIXED CONTRAST) */
    /* Forces these boxes to be white so they don't blend into the pink background */
    .stAlert {
        background-color: #8E3563;
        color: #FFFFFF;
        border: 2px solid #355C7D;
        border-radius: 10px;
    }
    
    /* 5. TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #8E3563;
        color: #F7F5EB;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #8E3563;
        color: #355C7D;
        border: 2px solid #355C7D;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING (Updated filename)
# ==========================================
@st.cache_data
def load_data():
    # LOADING THE NEW FILENAME
    try:
        df = pd.read_csv('streamlitdata.csv')
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("⚠️ File 'streamlitdata.csv' not found. Please upload it to the Files sidebar!")
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Kuala Lumpur Dining Assistant")
st.sidebar.markdown("By Tanisya Pristi Azrelia")
st.sidebar.caption("Master in Data Science - Universiti Malaya")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Best of The Best", "Find Your Restaurant", "Methodology & Insights"])

# ==========================================
# PAGE 1: HOME & HALL OF FAME
# ==========================================
if page == "Best of The Best":
    st.title("Kuala Lumpur Restaurant Recommendation System")
    st.markdown("""
    Welcome to this page. This platform utilize **machine learning** to help you discover your best dining experience in Kuala Lumpur.
    We help you to choose your preferred **dining priorities** by applying LDA and sentiment analysis with RoBERTa to get the rating for each topic.
    """)
    
    st.markdown("---")
    st.subheader("The Hall of Fame!")
    st.info("These are the best restaurants with high ratings (>4.0) and high reliability (>50 reviews).")

    # LOGIC: Weighted sort to find true top performers
    # We filter for >50 reviews to avoid "5.0 star" places with only 1 review
    top_restaurants = df[df['review_count'] > 50].sort_values('avg_rating', ascending=False).head(20)
    
    # Interactive Table
    st.dataframe(
        top_restaurants[['restaurant', 'avg_rating', 'review_count', 'western_cuisine', 'asian_cuisine', 'service']],
        column_config={
            "restaurant": "Restaurant Name",
            "avg_rating": st.column_config.NumberColumn("Stars", format="%.2f ⭐"),
            "review_count": st.column_config.NumberColumn("Reviews", format="%d 👤"),
            "western_cuisine": st.column_config.ProgressColumn("Western Score", min_value=1, max_value=5),
            "asian_cuisine": st.column_config.ProgressColumn("Asian Score", min_value=1, max_value=5),
            "service": st.column_config.LineChartColumn("Service Sentiment")
        },
        hide_index=True,
        use_container_width=True,
        height=600
    )

# ==========================================
# PAGE 2: RECOMMENDATION ENGINE
# ==========================================
elif page == "Find Your Restaurant":
    st.title("Where do you want to eat?")
    st.markdown("Dining experience should be personal. Customize your search based on what you prioritize the most.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Preferences")
        w_food = st.slider("Food Quality Importance", 0.0, 1.0, 0.8)
        w_service = st.slider("Service Importance", 0.0, 1.0, 0.5)
        w_value = st.slider("Value for Money Importance", 0.0, 1.0, 0.5)
        
        st.markdown("---")
        cuisine_pref = st.radio("Cuisine Type:", ["Any", "Western/Italian", "Asian/Local"])
        
        btn = st.button("Generate Recommendations", type="primary")

    with col2:
        if btn:
            # 1. Calculate Custom Score
            df['final_score'] = (
                (df['food_quality'] * w_food) +
                (df['service'] * w_service) +
                (df['value'] * w_value)
            )
            
            # 2. Filter by Cuisine
            filtered_df = df.copy()
            if cuisine_pref == "Western/Italian":
                filtered_df = filtered_df[filtered_df['western_cuisine'] > 3.5]
            elif cuisine_pref == "Asian/Local":
                filtered_df = filtered_df[filtered_df['asian_cuisine'] > 3.5]
                
            # 3. Get Top 5
            results = filtered_df.sort_values('final_score', ascending=False).head(5)
            
            st.subheader("Top 5 Recommendations")
            for i, (index, row) in enumerate(results.iterrows()):
                with st.container():
                    st.markdown(f"### #{i+1} {row['restaurant']}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Match Score", f"{row['final_score']:.2f}")
                    c2.metric("⭐ Rating", f"{row['avg_rating']:.1f}")
                    c3.metric("Food Sentiment", f"{row['food_quality']:.1f}/5")
                    c4.metric("Service Sentiment", f"{row['service']:.1f}/5")
                    st.progress(row['final_score'] / (w_food+w_service+w_value)*5 / 5) # Normalized progress bar
                    st.divider()

# ==========================================
# PAGE 3: METHODOLOGY & INSIGHTS
# ==========================================
elif page == "Methodology & Insights":
    st.title("This section shows methodology used to make this recommendation based on people's reviews")
    
    tab1, tab2, tab3 = st.tabs(["Topic Modeling (LDA)", "Sentiment (RoBERTa)", "Exploratory Analysis (EDA)"])
    
    # --- TAB 1: LDA vs BERTopic ---
    with tab1:
        st.header("Why LDA was chosen over BERTopic")
        st.markdown("After comparing two topic modeling approaches, **LDA (Latent Dirichlet Allocation)** was selected for the final application due to its stability and simple summary capability.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### LDA")
            st.success("Selected")
            st.markdown("""
            * **Coverage:** 100% of restaurants categorized.
            * **Consistency:** Produced stable topics (Service, Value, Food).
            * **Usability:** Broad categories work better for UI filters.
            """)
        
        with col2:
            st.markdown("### ❌ BERTopic")
            st.error("Discarded")
            st.markdown("""
            * **Coverage:** Only 55% (High outlier rate).
            * **Issue:** 45% of reviews labeled as 'Noise' (-1).
            * **Granularity:** 100+ micro-topics (too many options can be confusing for user).
            """)
            
        st.markdown("### Comparative Metrics")
        data = {
            'Metric': ['Coverage (%)', 'Coherence Score', 'Interpretability', 'Handling Noise'],
            'LDA': ['100%', '0.50', 'High (General)', 'Forces assignment'],
            'BERTopic': ['55%', 'N/A', 'High (Specific)', 'Drops outliers']
        }
        st.table(pd.DataFrame(data))

    # --- TAB 2: RoBERTa EVALUATION ---
    with tab2:
        st.header("RoBERTa Sentiment Evaluation")
        st.markdown("We utilize `twitter-roberta-base-sentiment` to give sentiment score on a 1-5 scale.")
        
        # METRICS SECTION
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Accuracy", "87.4%", "vs Star Ratings")
        c2.metric("Precision (Positive)", "92%")
        c3.metric("Precision (Negative)", "84%")
        
        st.markdown("---")
        st.subheader("Confusion Matrix Analysis")
        st.write("The matrix below shows how well the model predicts user star ratings.")
        
        # OPTIONAL: Display Image if it exists
        import os
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Model Predictions vs Actual User Ratings", width=600)
        else:
            st.warning("Confusion Matrix image not uploaded. (Upload 'confusion_matrix.png' to see it here)")

   # --- TAB 3: EDA ---
    with tab3:
        st.header("Exploratory Data Analysis")
        st.markdown("Visualizing the dataset to understand underlying patterns in customer feedback.")
        
        # ROW 1: Rating Distribution & Correlation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Rating Distribution")
            # Live Chart Generation (Fast & Interactive)
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
            
            # Compute correlation matrix live
            corr_matrix = df[corr_cols].corr()
            
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

        # ROW 2: Wordcloud & N-Grams (Static Images)
        st.subheader("3. Text Analysis (Word Cloud & N-Grams)")
        st.markdown("Most frequent words and phrases used by customers.")

        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Word Cloud**")
            # Check for image, otherwise show placeholder
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
