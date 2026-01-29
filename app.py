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
# 2. CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    /* Main background */
    .stApp { 
        background: linear-gradient(135deg, #355C7D 0%, #6C5B7B 50%, #C06C84 100%); 
        color: #FFFFFF; 
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 { 
        color: #FFFFFF !important; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5); 
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #F8B195 !important; font-size: 1.5rem !important; }
    div[data-testid="stMetricLabel"] { color: #FFFFFF !important; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { 
        background-color: #2A3E50; 
        color: #FFFFFF; 
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { 
        color: #F8B195 !important; 
    }
    
    /* Buttons */
    div.stButton > button { 
        background-color: #F8B195; 
        color: #355C7D; 
        font-weight: bold; 
        border: none; 
        border-radius: 10px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover { 
        background-color: #C06C84; 
        color: white; 
        transform: scale(1.05);
    }
    
    /* Cards */
    .restaurant-card {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #F8B195;
        backdrop-filter: blur(10px);
    }
    
    .rank-badge {
        background: linear-gradient(135deg, #F8B195, #C06C84);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-right: 10px;
    }
    
    .metric-box {
        background: rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
    
    .review-box {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-style: italic;
        border-left: 3px solid #F8B195;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: rgba(255,255,255,0.9) !important;
    }
    
    /* Winner badge */
    .winner-badge {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Table styling */
    .dataframe {
        background-color: rgba(255,255,255,0.9) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    """Load and clean the restaurant data"""
    filename = 'streamlitdata.csv'
    if not os.path.exists(filename):
        return None
    try:
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.error("CRITICAL ERROR: 'streamlitdata.csv' not found. Please upload it.")
    st.stop()

# ==========================================
# 4. DEFINE TOPIC COLUMNS
# ==========================================
TOPIC_COLUMNS = [
    'Ambiance & Atmosphere',
    'Staff Friendliness', 
    'Asian Cuisine',
    'Management',
    'Service Operations/Speed',
    'Western Cuisine',
    'Food Quality'
]

available_topics = [col for col in TOPIC_COLUMNS if col in df.columns]

# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================
def calculate_wlc_score(row, selected_topics):
    """
    Calculate Weighted Linear Combination (WLC) score
    Formula: Score = Σ(weight × aspect_score) / Σ(weights)
    """
    if not selected_topics:
        return row.get('avg_rating', 0)
    
    total_score = 0
    total_weight = 0
    
    for topic in selected_topics:
        if topic in row and pd.notna(row[topic]):
            weight = 1.0
            total_score += weight * row[topic]
            total_weight += weight
    
    if total_weight == 0:
        return row.get('avg_rating', 0)
    
    return total_score / total_weight

def get_sample_reviews(restaurant_name, num_reviews=3):
    """Get sample review texts for a restaurant"""
    if 'review' not in df.columns:
        return ["Review text not available in dataset"]
    
    restaurant_reviews = df[df['restaurant'] == restaurant_name]['review'].dropna()
    
    if len(restaurant_reviews) == 0:
        return ["No reviews available"]
    
    sample_size = min(num_reviews, len(restaurant_reviews))
    return restaurant_reviews.sample(sample_size).tolist()

# ==========================================
# 6. SIDEBAR
# ==========================================
st.sidebar.markdown("# KL Dining Assistant")
st.sidebar.markdown("**Topic-Based Restaurant Recommendation System**")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigate")

page = st.sidebar.radio(
    "",
    ["Best of The Best", "Find Your Restaurant", "Methodology & Insights"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Thesis Project**  
University of Malaya  
Master of Data Science  

By: **Tanisya Pristi Azrelia**  
ID: 24088031
""")

# ==========================================
# PAGE 1: BEST OF THE BEST
# ==========================================
if page == "Best of The Best":
    st.title("Best of The Best")
    st.markdown("### Top 20 Highest-Rated Restaurants in Kuala Lumpur")
    st.markdown("*Click on any restaurant to view customer reviews*")
    st.divider()
    
    if 'review_count' in df.columns and 'avg_rating' in df.columns:
        qualified_df = df[df['review_count'] >= 50].copy()
        
        if len(qualified_df) == 0:
            qualified_df = df.copy()
            st.warning("Showing all restaurants (none have 50+ reviews)")
        
        top_restaurants = qualified_df.groupby('restaurant').agg({
            'avg_rating': 'mean',
            'review_count': 'first'
        }).reset_index()
        
        top_restaurants = top_restaurants.sort_values('avg_rating', ascending=False).head(20)
        
        for idx, (_, row) in enumerate(top_restaurants.iterrows()):
            rank = idx + 1
            restaurant_name = row['restaurant']
            rating = row['avg_rating']
            review_count = row['review_count']
            
            rank_display = f"#{rank}"
            
            with st.expander(f"{rank_display} **{restaurant_name}** — {rating:.2f} ({int(review_count)} reviews)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Rating", f"{rating:.2f} / 5.0")
                with col2:
                    st.metric("Reviews", f"{int(review_count)}")
                with col3:
                    st.metric("Rank", f"#{rank}")
                
                st.markdown("---")
                st.markdown("#### Customer Reviews (Sample of 15)")
                
                reviews = get_sample_reviews(restaurant_name, num_reviews=15)
                
                for i, review in enumerate(reviews, 1):
                    if isinstance(review, str) and len(review) > 10:
                        display_review = review[:500] + "..." if len(review) > 500 else review
                        st.markdown(f"""
                        <div class="review-box">
                            <strong>Review {i}:</strong> "{display_review}"
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.error("Required columns 'avg_rating' or 'review_count' not found in data.")

# ==========================================
# PAGE 2: FIND YOUR RESTAURANT
# ==========================================
elif page == "Find Your Restaurant":
    st.title("Find Your Perfect Restaurant")
    st.markdown("### Personalized Recommendations Based on Your Preferences")
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Step 1: Select Your Priorities")
        st.markdown("*Choose what matters most to you (select 1 or more)*")
        
        if not available_topics:
            st.error("No topic columns found in the dataset!")
            st.write("Expected columns:", TOPIC_COLUMNS)
            st.write("Found columns:", list(df.columns))
            st.stop()
        
        selected_priorities = st.multiselect(
            "What aspects are important to you?",
            options=available_topics,
            default=None,
            help="Select one or more dining aspects. Recommendations will be ranked based on these priorities."
        )
        
        st.markdown("---")
        st.markdown("### Step 2: Cuisine Preference (Optional)")
        
        cuisine_filter = st.radio(
            "Filter by cuisine type:",
            options=["All Cuisines", "Western Only", "Asian Only"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### Step 3: Additional Filters")
        
        min_reviews = st.slider(
            "Minimum number of reviews:",
            min_value=1,
            max_value=100,
            value=10,
            help="Filter out restaurants with insufficient reviews"
        )
        
        min_rating = st.slider(
            "Minimum rating:",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5
        )
        
        st.markdown("---")
        find_button = st.button("Find Restaurants", type="primary", use_container_width=True)
    
    with col2:
        if find_button:
            if not selected_priorities:
                st.warning("Please select at least one priority to generate personalized recommendations.")
            else:
                st.markdown(f"### Top 10 Restaurants for: {', '.join(selected_priorities)}")
                
                filtered_df = df.copy()
                
                if cuisine_filter == "Western Only" and 'Western Cuisine' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Western Cuisine'] >= 3.0]
                elif cuisine_filter == "Asian Only" and 'Asian Cuisine' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Asian Cuisine'] >= 3.0]
                
                if 'review_count' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['review_count'] >= min_reviews]
                
                if 'avg_rating' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['avg_rating'] >= min_rating]
                
                filtered_df['match_score'] = filtered_df.apply(
                    lambda row: calculate_wlc_score(row, selected_priorities), 
                    axis=1
                )
                
                restaurant_scores = filtered_df.groupby('restaurant').agg({
                    'match_score': 'mean',
                    'avg_rating': 'mean',
                    'review_count': 'first'
                }).reset_index()
                
                top_10 = restaurant_scores.sort_values('match_score', ascending=False).head(10)
                
                if len(top_10) == 0:
                    st.warning("No restaurants match the specified criteria. Consider adjusting filters.")
                else:
                    for idx, (_, row) in enumerate(top_10.iterrows()):
                        rank = idx + 1
                        restaurant_name = row['restaurant']
                        match_score = row['match_score']
                        actual_rating = row['avg_rating']
                        review_count = row['review_count']
                        
                        st.markdown(f"""
                        <div class="restaurant-card">
                            <span class="rank-badge">#{rank}</span>
                            <strong style="font-size: 1.3rem;">{restaurant_name}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Match Score", f"{match_score:.2f} / 5.0")
                        with m2:
                            st.metric("Google Rating", f"{actual_rating:.2f} / 5.0")
                        with m3:
                            st.metric("Reviews", f"{int(review_count)}")
                        
                        with st.expander(f"View Customer Reviews"):
                            reviews = get_sample_reviews(restaurant_name, num_reviews=5)
                            for i, review in enumerate(reviews, 1):
                                if isinstance(review, str) and len(review) > 10:
                                    display_review = review[:400] + "..." if len(review) > 400 else review
                                    st.markdown(f'> **Review {i}:** "{display_review}"')
                        
                        st.markdown("---")
                    
                    with st.expander("How is Match Score calculated?"):
                        st.markdown("""
                        **Weighted Linear Combination (WLC) Formula:**
                        
                        ```
                        Match Score = Σ(weight × aspect_score) / Σ(weights)
                        ```
                        
                        With equal weights for selected preferences, the match score 
                        represents the arithmetic mean of the selected aspect scores.
                        
                        **Example Calculation:**
                        - Selected aspects: Food Quality, Service Operations/Speed
                        - Restaurant A scores: Food=4.5, Service=4.0
                        - Match Score = (4.5 + 4.0) / 2 = **4.25**
                        """)
        else:
            st.markdown("""
            ### System Overview
            
            1. **Select priorities** from the dropdown menu
            2. **Apply cuisine filter** if applicable
            3. **Set minimum thresholds** for reviews and ratings
            4. **Generate recommendations** based on specified criteria
            
            ---
            
            ### Machine Learning Pipeline
            
            This recommendation system integrates two machine learning components:
            
            **Topic Modeling (LDA)**
            - Latent Dirichlet Allocation extracts latent dining aspects from unstructured review text
            - Probabilistic model assigns topic distributions to each document
            
            **Sentiment Analysis (RoBERTa)**
            - Transformer-based model quantifies sentiment intensity for identified aspects
            - Pre-trained on 124 million tweets, fine-tuned for sentiment classification
            
            **Recommendation Algorithm (WLC)**
            - Weighted Linear Combination aggregates aspect-level sentiments
            - User-defined weights enable personalized ranking based on individual preferences
            """)

# ==========================================
# PAGE 3: METHODOLOGY & INSIGHTS
# ==========================================
elif page == "Methodology & Insights":
    st.title("Methodology & Insights")
    st.markdown("### Machine Learning Framework for Aspect-Based Recommendation")
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["LDA Topic Modeling", "RoBERTa Sentiment", "EDA Visualizations"])
    
    # ==========================================
    # TAB 1: LDA MODEL
    # ==========================================
    with tab1:
        st.markdown("## Latent Dirichlet Allocation (LDA)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Overview
            
            LDA is a generative probabilistic model for discovering latent topics 
            in document collections. The model assumes:
            
            - Each document is a mixture of topics
            - Each topic is a distribution over words
            - Documents are generated through a probabilistic process
            
            ### Model Selection Rationale
            
            LDA was selected over BERTopic based on comparative evaluation:
            
            | Criteria | LDA | BERTopic |
            |----------|-----|----------|
            | Coverage | **100%** | ~55% |
            | Topics | 7 (specified) | 81 (auto-generated) |
            | Outliers | **None** | 45% |
            | Interpretability | High | Moderate |
            
            BERTopic's HDBSCAN clustering generated 81 topics with 45% of documents 
            classified as outliers, rendering it unsuitable for the recommendation interface.
            """)
        
        with col2:
            st.markdown("### Discovered Topics (K=7)")
            
            topics_data = {
                'Topic': TOPIC_COLUMNS,
                'Description': [
                    'Restaurant atmosphere, decor, environment',
                    'Staff behavior, friendliness, hospitality',
                    'Asian cuisine quality and authenticity',
                    'Restaurant management and issue handling',
                    'Service speed, efficiency, operations',
                    'Western cuisine quality and preparation',
                    'Overall food quality, taste, presentation'
                ]
            }
            
            st.dataframe(
                pd.DataFrame(topics_data),
                hide_index=True,
                use_container_width=True
            )
            
            st.markdown("""
            <div class="winner-badge">
                LDA Selected: 100% Coverage with Interpretable Topics
            </div>
            """, unsafe_allow_html=True)
    
    # ==========================================
    # TAB 2: RoBERTa SENTIMENT
    # ==========================================
    with tab2:
        st.markdown("## RoBERTa Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Model Specification
            
            **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
            
            | Attribute | Value |
            |-----------|-------|
            | Developer | Cardiff NLP Group |
            | Architecture | RoBERTa-base |
            | Pre-training Data | ~58 million tweets |
            | Fine-tuning Data | ~124 million tweets |
            | Parameters | 125 million |
            | Output Classes | 3 (Negative, Neutral, Positive) |
            | Max Input Length | 512 tokens |
            
            ### Model Selection Rationale
            
            - Pre-trained on informal text similar to reviews
            - Handles colloquial expressions and abbreviations
            - No domain-specific fine-tuning required
            - Validated against ground-truth star ratings
            """)
        
        with col2:
            st.markdown("### Evaluation Metrics")
            
            metrics_data = {
                'Metric': ['Accuracy', 'Precision (Positive)', 'Precision (Negative)', 
                          'Recall (Positive)', 'Recall (Negative)', 'F1-Score'],
                'Score': ['87.03%', '93%', '62%', '97%', '84%', '85%']
            }
            
            st.dataframe(
                pd.DataFrame(metrics_data),
                hide_index=True,
                use_container_width=True
            )
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", "87.03%")
            m2.metric("F1-Score", "85%")
            m3.metric("Recall (Pos)", "97%")
        
        st.markdown("---")
        st.markdown("### Confusion Matrix")
        
        confusion_data = np.array([
            [84, 16],
            [3, 97]
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_data,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            text=confusion_data,
            texttemplate="%{text}%",
            textfont={"size": 20},
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title='Confusion Matrix (Normalized)',
            xaxis_title='Predicted Label',
            yaxis_title='Actual Label',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - 97% of positive reviews correctly classified
        - 84% of negative reviews correctly classified
        - Model exhibits slight bias toward positive predictions
        """)
    
    # ==========================================
    # TAB 3: EDA
    # ==========================================
    with tab3:
        st.markdown("## Exploratory Data Analysis")
        
        eda_tab1, eda_tab2 = st.tabs(["Word Cloud", "N-gram Analysis"])
        
        with eda_tab1:
            st.markdown("### Word Cloud Visualization")
            st.markdown("*Frequency distribution of terms in restaurant reviews*")
            
            wordcloud_paths = [
                'images/wordcloud.png',
                'figures/wordcloud.png',
                'wordcloud.png',
                'images/wordclouds.png',
                'figures/wordclouds.png'
            ]
            
            wordcloud_found = False
            for path in wordcloud_paths:
                if os.path.exists(path):
                    st.image(path, caption="Word Cloud of Restaurant Reviews", use_container_width=True)
                    wordcloud_found = True
                    break
            
            if not wordcloud_found:
                st.info("""
                Word Cloud image not found.
                
                Please upload to: `images/wordcloud.png`
                
                The word cloud visualizes term frequency distribution, 
                with larger terms indicating higher occurrence in the corpus.
                """)
        
        with eda_tab2:
            st.markdown("### N-gram Analysis")
            st.markdown("*Frequency distribution of word sequences in review corpus*")
            
            ngram_paths = [
                'images/ngram.png',
                'figures/ngram.png',
                'ngram.png',
                'images/ngram_analysis.png',
                'figures/ngram_analysis.png'
            ]
            
            ngram_found = False
            for path in ngram_paths:
                if os.path.exists(path):
                    st.image(path, caption="N-gram Analysis", use_container_width=True)
                    ngram_found = True
                    break
            
            if not ngram_found:
                st.info("""
                N-gram image not found.
                
                Please upload to: `images/ngram.png`
                
                N-gram analysis identifies frequently co-occurring word sequences,
                revealing common expressions and phrases in the review corpus.
                """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.7); padding: 1rem;">
    <p><strong>KL Dining Assistant</strong> | Topic-Based Restaurant Recommendation Platform</p>
    <p>LDA Topic Modeling & RoBERTa Sentiment Analysis</p>
    <p>Master of Data Science Thesis | University of Malaya | 2025</p>
    <p>Tanisya Pristi Azrelia (24088031)</p>
</div>
""", unsafe_allow_html=True)
