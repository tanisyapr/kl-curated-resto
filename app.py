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
    
    /* Headers - dark text for readability */
    h1 { 
        color: #5B7C99 !important; 
        font-weight: 700 !important;
    }
    h2, h3, h4, h5, h6 { 
        color: #7C5B7D !important; 
        font-weight: 600 !important;
    }
    
    /* Paragraphs and text */
    p, span, div, label {
        color: #2C3E50 !important;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { 
        color: #D4788C !important; 
        font-size: 1.5rem !important; 
        font-weight: bold !important;
    }
    div[data-testid="stMetricLabel"] { 
        color: #5B7C99 !important; 
        font-weight: 600 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #5B7C99 0%, #7C5B7D 100%);
    }
    section[data-testid="stSidebar"] * { 
        color: #FFFFFF !important; 
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #FFFFFF !important;
    }
    
    /* Buttons */
    div.stButton > button { 
        background: linear-gradient(135deg, #F4A5B8 0%, #A5C4D4 100%);
        color: #2C3E50 !important; 
        font-weight: bold; 
        border: none; 
        border-radius: 25px;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    div.stButton > button:hover { 
        background: linear-gradient(135deg, #D4788C 0%, #5B7C99 100%);
        color: #FFFFFF !important; 
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Restaurant cards */
    .restaurant-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F0F4 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #D4788C;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .rank-badge {
        background: linear-gradient(135deg, #D4788C, #A5C4D4);
        color: white !important;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-right: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .review-box {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #A5C4D4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        color: #2C3E50 !important;
    }
    
    .review-box strong {
        color: #5B7C99 !important;
    }
    
    /* Info boxes */
    .winner-badge {
        background: linear-gradient(135deg, #A5C4D4, #D4788C);
        color: white !important;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #F8F0F4 0%, #E8F4F8 100%) !important;
        border-radius: 10px !important;
        color: #2C3E50 !important;
        font-weight: 600 !important;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: #FFFFFF !important;
        border: 2px solid #A5C4D4 !important;
        border-radius: 10px !important;
    }
    
    /* Tables */
    .stDataFrame {
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F8F0F4;
        border-radius: 10px;
        color: #5B7C99 !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #D4788C, #A5C4D4) !important;
        color: white !important;
    }
    
    /* Divider */
    hr {
        border-color: #D4788C !important;
        opacity: 0.3;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #2C3E50 !important;
        font-weight: 500 !important;
    }
    
    /* Slider */
    .stSlider > div > div {
        color: #5B7C99 !important;
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
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.error("Data file 'streamlitdata.csv' not found. Please ensure the file is uploaded.")
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

def get_sample_reviews(restaurant_name, num_reviews=5):
    """Get sample review texts for a restaurant"""
    if 'review' not in df.columns:
        return []
    
    restaurant_data = df[df['restaurant'] == restaurant_name]
    if len(restaurant_data) == 0:
        return []
    
    review_text = restaurant_data['review'].iloc[0]
    if pd.isna(review_text) or review_text == '':
        return []
    
    # Split reviews by separator
    reviews = str(review_text).split(' | ')
    reviews = [r.strip() for r in reviews if r.strip() and r.strip() != 'No specific mentions.']
    
    return reviews[:num_reviews]

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
st.sidebar.markdown("""
**Thesis Project**  
University of Malaya  
Master of Data Science  

**Tanisya Pristi Azrelia**  
24088031
""")

# ==========================================
# PAGE 1: BEST OF THE BEST (Interactive Top 20)
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
            
            with st.expander(f"#{rank}  |  {restaurant_name}  |  Rating: {rating:.2f}  |  {int(review_count)} reviews"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Rating", f"{rating:.2f} / 5.0")
                with col2:
                    st.metric("Total Reviews", f"{int(review_count)}")
                with col3:
                    st.metric("Rank", f"#{rank}")
                
                st.markdown("---")
                st.markdown("#### Customer Reviews")
                
                reviews = get_sample_reviews(restaurant_name, num_reviews=10)
                
                if reviews:
                    for i, review in enumerate(reviews, 1):
                        if len(review) > 10:
                            display_review = review[:400] + "..." if len(review) > 400 else review
                            st.markdown(f"""
                            <div class="review-box">
                                <strong>Review {i}:</strong> {display_review}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No review text available for this restaurant.")
    else:
        st.error("Required columns 'avg_rating' or 'review_count' not found.")

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
        st.markdown("*Choose the aspects that matter most to you*")
        
        if not available_topics:
            st.error("No topic columns found in the dataset.")
            st.stop()
        
        # DROPDOWN MULTISELECT for topic selection
        selected_priorities = st.multiselect(
            "Select dining aspects (choose one or more):",
            options=available_topics,
            default=None,
            help="Select multiple aspects to find restaurants that excel in those areas."
        )
        
        st.markdown("---")
        st.markdown("### Step 2: Cuisine Filter (Optional)")
        
        cuisine_filter = st.selectbox(
            "Filter by cuisine type:",
            options=["All Cuisines", "Western Only", "Asian Only"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### Step 3: Quality Filters")
        
        min_reviews = st.slider(
            "Minimum reviews:",
            min_value=1,
            max_value=100,
            value=10
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
                st.warning("Please select at least one priority to generate recommendations.")
            else:
                st.markdown(f"### Top 10 Restaurants")
                st.markdown(f"**Selected Priorities:** {', '.join(selected_priorities)}")
                st.markdown("---")
                
                filtered_df = df.copy()
                
                # Apply cuisine filter
                if cuisine_filter == "Western Only" and 'Western Cuisine' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Western Cuisine'] >= 3.0]
                elif cuisine_filter == "Asian Only" and 'Asian Cuisine' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Asian Cuisine'] >= 3.0]
                
                # Apply review count filter
                if 'review_count' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['review_count'] >= min_reviews]
                
                # Apply rating filter
                if 'avg_rating' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['avg_rating'] >= min_rating]
                
                # Calculate WLC score
                filtered_df['match_score'] = filtered_df.apply(
                    lambda row: calculate_wlc_score(row, selected_priorities), 
                    axis=1
                )
                
                # Aggregate to restaurant level
                restaurant_scores = filtered_df.groupby('restaurant').agg({
                    'match_score': 'mean',
                    'avg_rating': 'mean',
                    'review_count': 'first'
                }).reset_index()
                
                # Get top 10
                top_10 = restaurant_scores.sort_values('match_score', ascending=False).head(10)
                
                if len(top_10) == 0:
                    st.warning("No restaurants match your criteria. Try adjusting the filters.")
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
                            <strong style="font-size: 1.4rem; color: #5B7C99;">{restaurant_name}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Match Score", f"{match_score:.2f}")
                        with m2:
                            st.metric("Google Rating", f"{actual_rating:.2f}")
                        with m3:
                            st.metric("Reviews", f"{int(review_count)}")
                        
                        # Show reviews
                        reviews = get_sample_reviews(restaurant_name, num_reviews=3)
                        if reviews:
                            with st.expander("View Customer Reviews"):
                                for i, review in enumerate(reviews, 1):
                                    if len(review) > 10:
                                        display_review = review[:300] + "..." if len(review) > 300 else review
                                        st.markdown(f"""
                                        <div class="review-box">
                                            <strong>Review {i}:</strong> {display_review}
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                    
                    # WLC explanation
                    with st.expander("How is Match Score calculated?"):
                        st.markdown("""
                        **Weighted Linear Combination (WLC) Formula:**
                        
                        ```
                        Match Score = Σ(weight × aspect_score) / Σ(weights)
                        ```
                        
                        With equal weights for selected preferences, the match score 
                        represents the arithmetic mean of the selected aspect scores.
                        
                        **Example:**
                        - Selected: Food Quality, Service Operations/Speed
                        - Restaurant A: Food = 4.5, Service = 4.0
                        - Match Score = (4.5 + 4.0) / 2 = **4.25**
                        """)
        else:
            st.markdown("""
            ### How to Use
            
            1. **Select your priorities** from the dropdown menu on the left
            2. **Choose cuisine type** if you have a preference
            3. **Set minimum thresholds** for reviews and ratings
            4. **Click "Find Restaurants"** to view personalized recommendations
            
            ---
            
            ### Machine Learning Pipeline
            
            This system combines two machine learning approaches:
            
            **Topic Modeling (LDA)**  
            Latent Dirichlet Allocation extracts dining aspects from review text, 
            identifying what customers discuss about each restaurant.
            
            **Sentiment Analysis (RoBERTa)**  
            A transformer-based model quantifies sentiment for each aspect, 
            converting text into numerical scores.
            
            **Recommendation (WLC)**  
            Weighted Linear Combination ranks restaurants based on your 
            selected priorities.
            """)

# ==========================================
# PAGE 3: METHODOLOGY & INSIGHTS
# ==========================================
elif page == "Methodology & Insights":
    st.title("Methodology & Insights")
    st.markdown("### Machine Learning Framework for Aspect-Based Recommendation")
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["LDA Topic Modeling", "RoBERTa Sentiment", "EDA"])
    
    # TAB 1: LDA
    with tab1:
        st.markdown("## Latent Dirichlet Allocation (LDA)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Overview
            
            LDA is a generative probabilistic model for discovering latent topics 
            in document collections.
            
            **Key Assumptions:**
            - Each document is a mixture of topics
            - Each topic is a distribution over words
            - Topics are discovered through probabilistic inference
            
            ### Model Comparison
            
            | Criteria | LDA | BERTopic |
            |----------|-----|----------|
            | Coverage | **100%** | ~55% |
            | Topics | 7 (specified) | 81 (auto) |
            | Outliers | **None** | 45% |
            | Interpretability | High | Moderate |
            
            BERTopic generated 81 topics with 45% outliers, making it 
            unsuitable for the recommendation interface.
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
    
    # TAB 2: RoBERTa
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
            | Pre-training | ~58M tweets |
            | Fine-tuning | ~124M tweets |
            | Parameters | 125 million |
            | Output | 3 classes |
            | Max Length | 512 tokens |
            
            ### Selection Rationale
            - Pre-trained on informal text (similar to reviews)
            - Handles colloquial language effectively
            - No domain-specific fine-tuning required
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
            m3.metric("Recall+", "97%")
        
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
            textfont={"size": 18, "color": "white"},
            colorscale=[[0, '#A5C4D4'], [1, '#D4788C']],
            showscale=False
        ))
        
        fig.update_layout(
            title='Confusion Matrix (Normalized %)',
            xaxis_title='Predicted Label',
            yaxis_title='Actual Label',
            height=350,
            font=dict(color='#2C3E50')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - 97% of positive reviews correctly classified
        - 84% of negative reviews correctly classified
        """)
    
    # TAB 3: EDA (Wordcloud & Ngram only)
    with tab3:
        st.markdown("## Exploratory Data Analysis")
        
        eda_col1, eda_col2 = st.columns(2)
        
        with eda_col1:
            st.markdown("### Word Cloud")
            st.markdown("*Term frequency distribution in review corpus*")
            
            wordcloud_paths = ['images/wordcloud.png', 'figures/wordcloud.png', 'wordcloud.png']
            wordcloud_found = False
            
            for path in wordcloud_paths:
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                    wordcloud_found = True
                    break
            
            if not wordcloud_found:
                st.info("Upload wordcloud image to: images/wordcloud.png")
        
        with eda_col2:
            st.markdown("### N-gram Analysis")
            st.markdown("*Frequent word sequences in reviews*")
            
            ngram_paths = ['images/ngram.png', 'figures/ngram.png', 'ngram.png']
            ngram_found = False
            
            for path in ngram_paths:
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                    ngram_found = True
                    break
            
            if not ngram_found:
                st.info("Upload ngram image to: images/ngram.png")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #5B7C99; padding: 1rem;">
    <p><strong>KL Dining Assistant</strong> | Topic-Based Restaurant Recommendation Platform</p>
    <p>LDA Topic Modeling & RoBERTa Sentiment Analysis</p>
    <p>Master of Data Science Thesis | University of Malaya | 2025</p>
    <p>Tanisya Pristi Azrelia (24088031)</p>
</div>
""", unsafe_allow_html=True)
