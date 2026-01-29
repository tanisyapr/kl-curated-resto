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
    p, span, label {
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
    
    /* MULTISELECT & DROPDOWN - HIGH CONTRAST FIX */
    .stMultiSelect > div > div {
        background-color: #FFFFFF !important;
        border: 2px solid #5B7C99 !important;
        border-radius: 10px !important;
    }
    .stMultiSelect span {
        color: #2C3E50 !important;
    }
    .stMultiSelect div[data-baseweb="tag"] {
        background-color: #D4788C !important;
        color: #FFFFFF !important;
    }
    .stMultiSelect div[data-baseweb="tag"] span {
        color: #FFFFFF !important;
    }
    .stMultiSelect svg {
        fill: #FFFFFF !important;
    }
    
    /* Selectbox dropdown */
    .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        border: 2px solid #5B7C99 !important;
        border-radius: 10px !important;
        color: #2C3E50 !important;
    }
    .stSelectbox label {
        color: #2C3E50 !important;
        font-weight: 600 !important;
    }
    
    /* Dropdown menu items */
    div[data-baseweb="popover"] {
        background-color: #FFFFFF !important;
    }
    div[data-baseweb="popover"] li {
        color: #2C3E50 !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #F8E8F0 !important;
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
    
    /* Winner badge */
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
    
    /* TOP 20 CARDS - COLORFUL */
    .top-card-gold {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 6px 20px rgba(255,215,0,0.3);
        border: none;
    }
    .top-card-silver {
        background: linear-gradient(135deg, #E8E8E8 0%, #C0C0C0 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 6px 20px rgba(192,192,192,0.3);
        border: none;
    }
    .top-card-bronze {
        background: linear-gradient(135deg, #E8C4A0 0%, #CD7F32 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 6px 20px rgba(205,127,50,0.3);
        border: none;
    }
    .top-card-regular {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F8FF 100%);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.6rem 0;
        border-left: 4px solid #A5C4D4;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    
    .rank-gold { color: #8B6914 !important; font-size: 2rem !important; font-weight: bold !important; }
    .rank-silver { color: #5A5A5A !important; font-size: 1.8rem !important; font-weight: bold !important; }
    .rank-bronze { color: #8B4513 !important; font-size: 1.6rem !important; font-weight: bold !important; }
    .rank-regular { color: #5B7C99 !important; font-size: 1.2rem !important; font-weight: bold !important; }
    
    .restaurant-name-top {
        font-size: 1.4rem !important;
        font-weight: bold !important;
        color: #2C3E50 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #F8F0F4 0%, #E8F4F8 100%) !important;
        border-radius: 10px !important;
        color: #2C3E50 !important;
        font-weight: 600 !important;
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
    
    /* Radio buttons */
    .stRadio > label {
        color: #2C3E50 !important;
        font-weight: 500 !important;
    }
    .stRadio div[role="radiogroup"] label {
        color: #2C3E50 !important;
    }
    
    /* Slider labels */
    .stSlider label {
        color: #2C3E50 !important;
        font-weight: 600 !important;
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
# 4. DEFINE TOPIC COLUMNS (YOUR ACTUAL DATA)
# ==========================================
TOPIC_COLUMNS = [
    'FOOD QUALITY',
    'LOCATION',
    'SERVICE',
    'Topic 5',
    'VALUE'
]

# User-friendly labels for display
TOPIC_LABELS = {
    'FOOD QUALITY': 'Food Quality',
    'LOCATION': 'Location & Accessibility',
    'SERVICE': 'Service Quality',
    'Topic 5': 'Dining Experience',
    'VALUE': 'Value for Money'
}

available_topics = [col for col in TOPIC_COLUMNS if col in df.columns]

# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================
def calculate_wlc_score(row, selected_topics):
    """
    Calculate Weighted Linear Combination (WLC) score
    Formula: Score = Σ(weight × aspect_score) / Σ(weights)
    Ensures score is within 1-5 range
    """
    if not selected_topics:
        rating = row.get('avg_rating', 3.0)
        return min(max(float(rating), 1.0), 5.0)
    
    total_score = 0
    total_weight = 0
    
    for topic in selected_topics:
        if topic in row and pd.notna(row[topic]):
            score = float(row[topic])
            # Ensure score is within valid range
            score = min(max(score, 1.0), 5.0)
            weight = 1.0
            total_score += weight * score
            total_weight += weight
    
    if total_weight == 0:
        rating = row.get('avg_rating', 3.0)
        return min(max(float(rating), 1.0), 5.0)
    
    final_score = total_score / total_weight
    # Ensure final score is within 1-5
    return min(max(final_score, 1.0), 5.0)

def get_sample_reviews(restaurant_name, num_reviews=5):
    """Get sample review texts for a restaurant from text columns"""
    restaurant_data = df[df['restaurant'] == restaurant_name]
    if len(restaurant_data) == 0:
        return []
    
    row = restaurant_data.iloc[0]
    reviews = []
    
    # Collect from all text columns
    text_columns = ['FOOD QUALITY_text', 'LOCATION_text', 'SERVICE_text', 'Topic 5_text', 'VALUE_text']
    
    for col in text_columns:
        if col in row and pd.notna(row[col]):
            text = str(row[col])
            if text and text != 'nan' and text != 'No specific mentions.' and len(text) > 10:
                # Split if multiple reviews in one cell
                parts = text.split(' | ')
                for part in parts:
                    if part.strip() and part.strip() != 'No specific mentions.' and len(part.strip()) > 10:
                        reviews.append(part.strip())
    
    return reviews[:num_reviews]

def format_rating(rating):
    """Format rating to be within 1-5 scale"""
    try:
        r = float(rating)
        r = min(max(r, 1.0), 5.0)
        return f"{r:.2f}"
    except:
        return "N/A"

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
# PAGE 1: BEST OF THE BEST (Colorful & Interactive)
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
            st.info("Showing all restaurants")
        
        # Ensure rating is within valid range
        qualified_df['avg_rating'] = qualified_df['avg_rating'].apply(lambda x: min(max(float(x), 1.0), 5.0))
        
        top_restaurants = qualified_df.groupby('restaurant').agg({
            'avg_rating': 'mean',
            'review_count': 'first'
        }).reset_index()
        
        # Ensure aggregated rating is within range
        top_restaurants['avg_rating'] = top_restaurants['avg_rating'].apply(lambda x: min(max(x, 1.0), 5.0))
        
        top_restaurants = top_restaurants.sort_values('avg_rating', ascending=False).head(20)
        
        for idx, (_, row) in enumerate(top_restaurants.iterrows()):
            rank = idx + 1
            restaurant_name = row['restaurant']
            rating = min(max(row['avg_rating'], 1.0), 5.0)  # Ensure 1-5 range
            review_count = int(row['review_count'])
            
            # Different styling for top 3
            if rank == 1:
                card_class = "top-card-gold"
                rank_class = "rank-gold"
                rank_emoji = "🥇"
            elif rank == 2:
                card_class = "top-card-silver"
                rank_class = "rank-silver"
                rank_emoji = "🥈"
            elif rank == 3:
                card_class = "top-card-bronze"
                rank_class = "rank-bronze"
                rank_emoji = "🥉"
            else:
                card_class = "top-card-regular"
                rank_class = "rank-regular"
                rank_emoji = f"#{rank}"
            
            with st.expander(f"{rank_emoji}  {restaurant_name}  —  {rating:.2f}/5.0  ({review_count} reviews)", expanded=(rank <= 3)):
                st.markdown(f"""
                <div class="{card_class}">
                    <span class="{rank_class}">{rank_emoji}</span>
                    <span class="restaurant-name-top">{restaurant_name}</span>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rating", f"{rating:.2f} / 5.0")
                with col2:
                    st.metric("Total Reviews", f"{review_count}")
                with col3:
                    st.metric("Rank", f"#{rank}")
                
                st.markdown("---")
                st.markdown("#### Customer Reviews")
                
                reviews = get_sample_reviews(restaurant_name, num_reviews=8)
                
                if reviews:
                    for i, review in enumerate(reviews, 1):
                        display_review = review[:350] + "..." if len(review) > 350 else review
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
        
        # Create display options with friendly labels
        display_options = [TOPIC_LABELS.get(t, t) for t in available_topics]
        label_to_column = {TOPIC_LABELS.get(t, t): t for t in available_topics}
        
        # DROPDOWN MULTISELECT
        selected_display = st.multiselect(
            "Select dining aspects (choose one or more):",
            options=display_options,
            default=None,
            help="Select multiple aspects to find restaurants that excel in those areas."
        )
        
        # Convert back to actual column names
        selected_priorities = [label_to_column[d] for d in selected_display]
        
        st.markdown("---")
        st.markdown("### Step 2: Quality Filters")
        
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
                selected_labels = [TOPIC_LABELS.get(p, p) for p in selected_priorities]
                st.markdown(f"**Selected Priorities:** {', '.join(selected_labels)}")
                st.markdown("---")
                
                filtered_df = df.copy()
                
                # Apply review count filter
                if 'review_count' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['review_count'] >= min_reviews]
                
                # Apply rating filter (ensure rating is in valid range first)
                if 'avg_rating' in filtered_df.columns:
                    filtered_df['avg_rating'] = filtered_df['avg_rating'].apply(lambda x: min(max(float(x), 1.0), 5.0))
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
                
                # Ensure scores are in valid range
                restaurant_scores['match_score'] = restaurant_scores['match_score'].apply(lambda x: min(max(x, 1.0), 5.0))
                restaurant_scores['avg_rating'] = restaurant_scores['avg_rating'].apply(lambda x: min(max(x, 1.0), 5.0))
                
                # Get top 10
                top_10 = restaurant_scores.sort_values('match_score', ascending=False).head(10)
                
                if len(top_10) == 0:
                    st.warning("No restaurants match your criteria. Try adjusting the filters.")
                else:
                    for idx, (_, row) in enumerate(top_10.iterrows()):
                        rank = idx + 1
                        restaurant_name = row['restaurant']
                        match_score = min(max(row['match_score'], 1.0), 5.0)
                        actual_rating = min(max(row['avg_rating'], 1.0), 5.0)
                        review_count = int(row['review_count'])
                        
                        st.markdown(f"""
                        <div class="restaurant-card">
                            <span class="rank-badge">#{rank}</span>
                            <strong style="font-size: 1.4rem; color: #5B7C99;">{restaurant_name}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Match Score", f"{match_score:.2f} / 5.0")
                        with m2:
                            st.metric("Google Rating", f"{actual_rating:.2f} / 5.0")
                        with m3:
                            st.metric("Reviews", f"{review_count}")
                        
                        # Show reviews
                        reviews = get_sample_reviews(restaurant_name, num_reviews=3)
                        if reviews:
                            with st.expander("View Customer Reviews"):
                                for i, review in enumerate(reviews, 1):
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
                        - Selected: Food Quality, Service Quality
                        - Restaurant A: Food = 4.5, Service = 4.0
                        - Match Score = (4.5 + 4.0) / 2 = **4.25**
                        """)
        else:
            st.markdown("""
            ### How to Use
            
            1. **Select your priorities** from the dropdown menu
            2. **Set minimum thresholds** for reviews and ratings
            3. **Click "Find Restaurants"** to view recommendations
            
            ---
            
            ### Machine Learning Pipeline
            
            This system combines two machine learning approaches:
            
            **Topic Modeling (LDA)**  
            Latent Dirichlet Allocation extracts dining aspects from review text.
            
            **Sentiment Analysis (RoBERTa)**  
            A transformer-based model quantifies sentiment for each aspect.
            
            **Recommendation (WLC)**  
            Weighted Linear Combination ranks restaurants based on your priorities.
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
            | Topics | 5 (specified) | 81 (auto) |
            | Outliers | **None** | 45% |
            | Interpretability | High | Moderate |
            
            BERTopic generated 81 topics with 45% outliers, making it 
            unsuitable for the recommendation interface.
            """)
        
        with col2:
            st.markdown("### Discovered Topics (K=5)")
            
            topics_data = {
                'Topic': ['Food Quality', 'Location', 'Service', 'Topic 5', 'Value'],
                'Description': [
                    'Overall food quality, taste, presentation',
                    'Restaurant location and accessibility',
                    'Service quality and staff behavior',
                    'General dining experience',
                    'Value for money and pricing'
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
            - Pre-trained on informal text
            - Handles colloquial language
            - No fine-tuning required
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
    
    # TAB 3: EDA (Wordcloud & Ngram only)
    with tab3:
        st.markdown("## Exploratory Data Analysis")
        
        eda_col1, eda_col2 = st.columns(2)
        
        with eda_col1:
            st.markdown("### Word Cloud")
            st.markdown("*Term frequency distribution*")
            
            wordcloud_paths = ['images/wordcloud.png', 'figures/wordcloud.png', 'wordcloud.png']
            wordcloud_found = False
            
            for path in wordcloud_paths:
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                    wordcloud_found = True
                    break
            
            if not wordcloud_found:
                st.info("Upload wordcloud to: images/wordcloud.png")
        
        with eda_col2:
            st.markdown("### N-gram Analysis")
            st.markdown("*Frequent word sequences*")
            
            ngram_paths = ['images/ngram.png', 'figures/ngram.png', 'ngram.png']
            ngram_found = False
            
            for path in ngram_paths:
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                    ngram_found = True
                    break
            
            if not ngram_found:
                st.info("Upload ngram to: images/ngram.png")

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
