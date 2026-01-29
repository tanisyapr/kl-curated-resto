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
    st.error("⚠️ CRITICAL ERROR: 'streamlitdata.csv' not found. Please upload it.")
    st.stop()

# ==========================================
# 4. DEFINE TOPIC COLUMNS
# ==========================================
# Your 7 topics from LDA
TOPIC_COLUMNS = [
    'Ambiance & Atmosphere',
    'Staff Friendliness', 
    'Asian Cuisine',
    'Management',
    'Service Operation/Speed',
    'Western Cuisine',
    'Food Quality'
]

# Check which topics exist in the dataframe
available_topics = [col for col in TOPIC_COLUMNS if col in df.columns]

# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================
def calculate_wlc_score(row, selected_topics):
    """
    Calculate Weighted Linear Combination (WLC) score
    
    Formula: Score = Σ(weight × aspect_score) / Σ(weights)
    
    Since user selects topics (no explicit weights), we use equal weights (1.0)
    """
    if not selected_topics:
        return row.get('avg_rating', 0)
    
    total_score = 0
    total_weight = 0
    
    for topic in selected_topics:
        if topic in row and pd.notna(row[topic]):
            weight = 1.0  # Equal weights for selected topics
            total_score += weight * row[topic]
            total_weight += weight
    
    if total_weight == 0:
        return row.get('avg_rating', 0)
    
    return total_score / total_weight

def get_star_display(rating):
    """Convert rating to star emoji display"""
    full_stars = int(rating)
    half_star = 1 if rating - full_stars >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    return "⭐" * full_stars + "✨" * half_star + "☆" * empty_stars

def get_sample_reviews(restaurant_name, num_reviews=3):
    """Get sample review texts for a restaurant"""
    # Check if 'review' column exists
    if 'review' not in df.columns:
        return ["Review text not available in dataset"]
    
    restaurant_reviews = df[df['restaurant'] == restaurant_name]['review'].dropna()
    
    if len(restaurant_reviews) == 0:
        return ["No reviews available"]
    
    # Get random sample of reviews
    sample_size = min(num_reviews, len(restaurant_reviews))
    return restaurant_reviews.sample(sample_size).tolist()

# ==========================================
# 6. SIDEBAR
# ==========================================
st.sidebar.markdown("# 🍽️ KL Dining Assistant")
st.sidebar.markdown("**Topic-Based Restaurant Recommendation System**")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Navigate")

page = st.sidebar.radio(
    "",
    ["🏆 Best of The Best", "🔍 Find Your Restaurant", "📊 Methodology & Insights"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
if 'restaurant' in df.columns:
    unique_restaurants = df['restaurant'].nunique()
    st.sidebar.metric("Restaurants", unique_restaurants)
if 'review_count' in df.columns:
    total_reviews = df['review_count'].sum()
    st.sidebar.metric("Total Reviews", f"{int(total_reviews):,}")
if 'avg_rating' in df.columns:
    avg_rating = df['avg_rating'].mean()
    st.sidebar.metric("Avg Rating", f"{avg_rating:.2f} ⭐")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👩‍💻 About")
st.sidebar.info("""
**Thesis Project**  
University of Malaya  
Master of Data Science  

By: **Tanisya Pristi Azrelia**  
ID: 24088031
""")

# ==========================================
# PAGE 1: BEST OF THE BEST (Interactive Top 20)
# ==========================================
if page == "🏆 Best of The Best":
    st.title("🏆 Best of The Best")
    st.markdown("### Top 20 Highest-Rated Restaurants in Kuala Lumpur")
    st.markdown("*Click on any restaurant to see customer reviews!*")
    st.divider()
    
    # Get Top 20 restaurants
    if 'review_count' in df.columns and 'avg_rating' in df.columns:
        # Filter restaurants with sufficient reviews (>50 for reliability)
        qualified_df = df[df['review_count'] >= 50].copy()
        
        if len(qualified_df) == 0:
            qualified_df = df.copy()
            st.warning("Showing all restaurants (none have 50+ reviews)")
        
        # Get unique restaurants with their best scores
        top_restaurants = qualified_df.groupby('restaurant').agg({
            'avg_rating': 'mean',
            'review_count': 'first'
        }).reset_index()
        
        top_restaurants = top_restaurants.sort_values('avg_rating', ascending=False).head(20)
        
        # Display as interactive cards
        for idx, (_, row) in enumerate(top_restaurants.iterrows()):
            rank = idx + 1
            restaurant_name = row['restaurant']
            rating = row['avg_rating']
            review_count = row['review_count']
            
            # Rank emoji
            if rank == 1:
                rank_display = "🥇"
            elif rank == 2:
                rank_display = "🥈"
            elif rank == 3:
                rank_display = "🥉"
            else:
                rank_display = f"#{rank}"
            
            # Create expander for each restaurant
            with st.expander(f"{rank_display} **{restaurant_name}** — ⭐ {rating:.2f} ({int(review_count)} reviews)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Rating", f"{rating:.2f} / 5.0")
                with col2:
                    st.metric("Reviews", f"{int(review_count)}")
                with col3:
                    st.metric("Rank", f"#{rank}")
                
                st.markdown("---")
                st.markdown("#### 💬 Customer Reviews (Sample of 15)")
                
                # Get 15 sample reviews
                reviews = get_sample_reviews(restaurant_name, num_reviews=15)
                
                for i, review in enumerate(reviews, 1):
                    if isinstance(review, str) and len(review) > 10:
                        # Truncate very long reviews
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
elif page == "🔍 Find Your Restaurant":
    st.title("🔍 Find Your Perfect Restaurant")
    st.markdown("### Personalized Recommendations Based on Your Preferences")
    st.divider()
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎯 Step 1: Select Your Priorities")
        st.markdown("*Choose what matters most to you (select 1 or more)*")
        
        # Check if topics are available
        if not available_topics:
            st.error("⚠️ No topic columns found in the dataset!")
            st.write("Expected columns:", TOPIC_COLUMNS)
            st.write("Found columns:", list(df.columns))
            st.stop()
        
        # Multi-select dropdown for topics
        selected_priorities = st.multiselect(
            "What aspects are important to you?",
            options=available_topics,
            default=None,
            help="Select one or more dining aspects. Your recommendations will be ranked based on these priorities."
        )
        
        st.markdown("---")
        st.markdown("### 🍜 Step 2: Cuisine Preference (Optional)")
        
        cuisine_filter = st.radio(
            "Filter by cuisine type:",
            options=["All Cuisines", "Western Only", "Asian Only"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Step 3: Additional Filters")
        
        min_reviews = st.slider(
            "Minimum number of reviews:",
            min_value=1,
            max_value=100,
            value=10,
            help="Filter out restaurants with too few reviews for reliability"
        )
        
        min_rating = st.slider(
            "Minimum rating:",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5
        )
        
        st.markdown("---")
        find_button = st.button("🔍 Find My Restaurants", type="primary", use_container_width=True)
    
    with col2:
        if find_button:
            # Validate selection
            if not selected_priorities:
                st.warning("⚠️ Please select at least one priority to get personalized recommendations!")
                st.info("💡 Tip: Select aspects like 'Food Quality', 'Service Operation/Speed', etc.")
            else:
                st.markdown(f"### 🎉 Top 10 Restaurants for: {', '.join(selected_priorities)}")
                
                # Calculate WLC scores
                filtered_df = df.copy()
                
                # Apply cuisine filter
                if cuisine_filter == "Western Only" and 'Western Cuisine' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Western Cuisine'] >= 3.0]
                elif cuisine_filter == "Asian Only" and 'Asian Cuisine' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Asian Cuisine'] >= 3.0]
                
                # Apply minimum reviews filter
                if 'review_count' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['review_count'] >= min_reviews]
                
                # Apply minimum rating filter
                if 'avg_rating' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['avg_rating'] >= min_rating]
                
                # Calculate WLC score for each row
                filtered_df['match_score'] = filtered_df.apply(
                    lambda row: calculate_wlc_score(row, selected_priorities), 
                    axis=1
                )
                
                # Aggregate to restaurant level (in case of multiple rows per restaurant)
                restaurant_scores = filtered_df.groupby('restaurant').agg({
                    'match_score': 'mean',
                    'avg_rating': 'mean',
                    'review_count': 'first'
                }).reset_index()
                
                # Sort and get top 10
                top_10 = restaurant_scores.sort_values('match_score', ascending=False).head(10)
                
                if len(top_10) == 0:
                    st.warning("No restaurants match your criteria. Try adjusting your filters!")
                else:
                    # Display results
                    for idx, (_, row) in enumerate(top_10.iterrows()):
                        rank = idx + 1
                        restaurant_name = row['restaurant']
                        match_score = row['match_score']
                        actual_rating = row['avg_rating']
                        review_count = row['review_count']
                        
                        # Create card for each restaurant
                        st.markdown(f"""
                        <div class="restaurant-card">
                            <span class="rank-badge">#{rank}</span>
                            <strong style="font-size: 1.3rem;">{restaurant_name}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics row
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("🎯 Match Score", f"{match_score:.2f} / 5.0")
                        with m2:
                            st.metric("⭐ Google Rating", f"{actual_rating:.2f} / 5.0")
                        with m3:
                            st.metric("📝 Reviews", f"{int(review_count)}")
                        
                        # Show sample reviews
                        with st.expander(f"💬 See Customer Reviews"):
                            reviews = get_sample_reviews(restaurant_name, num_reviews=5)
                            for i, review in enumerate(reviews, 1):
                                if isinstance(review, str) and len(review) > 10:
                                    display_review = review[:400] + "..." if len(review) > 400 else review
                                    st.markdown(f'> **Review {i}:** "{display_review}"')
                        
                        st.markdown("---")
                    
                    # Show WLC explanation
                    with st.expander("ℹ️ How is Match Score calculated?"):
                        st.markdown("""
                        **Weighted Linear Combination (WLC) Formula:**
                        
                        ```
                        Match Score = Σ(weight × aspect_score) / Σ(weights)
                        ```
                        
                        Since you selected multiple preferences with equal importance, 
                        the match score is the **average** of your selected aspect scores.
                        
                        **Example:**
                        - You selected: Food Quality, Service Operation/Speed
                        - Restaurant A: Food=4.5, Service=4.0
                        - Match Score = (4.5 + 4.0) / 2 = **4.25**
                        """)
        else:
            # Show instructions when button not clicked
            st.markdown("""
            ### 📋 How It Works
            
            1. **Select your priorities** from the dropdown on the left
            2. **Choose cuisine preference** if you have one
            3. **Adjust filters** for minimum reviews and rating
            4. **Click "Find My Restaurants"** to see personalized recommendations!
            
            ---
            
            ### 🧠 The Science Behind It
            
            Our recommendation engine uses:
            - **LDA Topic Modeling** to identify dining aspects
            - **RoBERTa Sentiment Analysis** to score each aspect
            - **Weighted Linear Combination** to rank restaurants based on YOUR priorities
            
            This means two restaurants with the same overall rating might rank 
            differently depending on what YOU care about!
            """)

# ==========================================
# PAGE 3: METHODOLOGY & INSIGHTS
# ==========================================
elif page == "📊 Methodology & Insights":
    st.title("📊 Methodology & Insights")
    st.markdown("### Understanding the Science Behind the Recommendations")
    st.divider()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 LDA Topic Modeling", "🤖 RoBERTa Sentiment", "📈 EDA Visualizations"])
    
    # ==========================================
    # TAB 1: LDA MODEL
    # ==========================================
    with tab1:
        st.markdown("## Latent Dirichlet Allocation (LDA)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### What is LDA?
            
            LDA is a **probabilistic topic modeling** algorithm that discovers 
            hidden topics in a collection of documents.
            
            **Key Characteristics:**
            - Each review is a **mixture of topics**
            - Each topic is a **distribution over words**
            - Achieves **100% data coverage**
            
            ### Why LDA over BERTopic?
            
            | Criteria | LDA | BERTopic |
            |----------|-----|----------|
            | Coverage | **100%** | ~55% |
            | Topics | 7 (controlled) | 100+ (uncontrolled) |
            | Outliers | **None** | 45% |
            | UI Practical | ✅ Yes | ❌ No |
            """)
        
        with col2:
            st.markdown("### Discovered Topics (K=7)")
            
            topics_data = {
                'Topic': TOPIC_COLUMNS,
                'Description': [
                    'Restaurant atmosphere, decor, ambiance',
                    'Staff behavior, friendliness, hospitality',
                    'Asian cuisine quality and authenticity',
                    'Restaurant management, handling issues',
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
                🏆 LDA Selected for 100% Coverage & Interpretability
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
            ### Model Information
            
            **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
            
            | Attribute | Value |
            |-----------|-------|
            | Developer | Cardiff NLP (Cardiff University) |
            | Pre-training | ~58 million tweets |
            | Fine-tuning | ~124 million tweets |
            | Parameters | 125 million |
            | Output | 3 classes (Neg, Neu, Pos) |
            | Max Input | 512 tokens |
            
            ### Why This Model?
            - Trained on informal text (similar to reviews)
            - Handles slang, abbreviations, emoticons
            - State-of-the-art performance
            - No fine-tuning needed
            """)
        
        with col2:
            st.markdown("### Evaluation Metrics")
            
            # Metrics display
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
            
            # Key metrics as visual metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", "87.03%", delta="Good")
            m2.metric("F1-Score", "85%", delta="Good")
            m3.metric("Recall (Pos)", "97%", delta="Excellent")
        
        st.markdown("---")
        st.markdown("### Confusion Matrix")
        
        # Create confusion matrix visualization
        confusion_data = np.array([
            [84, 16],   # Actual Negative: 84 correct, 16 wrong
            [3, 97]     # Actual Positive: 3 wrong, 97 correct
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_data,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            text=confusion_data,
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title='Confusion Matrix (Normalized %)',
            xaxis_title='Predicted Label',
            yaxis_title='Actual Label',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - **97%** of positive reviews correctly identified
        - **84%** of negative reviews correctly identified
        - Model tends to slightly over-predict positive sentiment
        """)
    
    # ==========================================
    # TAB 3: EDA - WORDCLOUD & NGRAM
    # ==========================================
    with tab3:
        st.markdown("## Exploratory Data Analysis")
        
        eda_tab1, eda_tab2 = st.tabs(["☁️ Word Cloud", "📊 N-gram Analysis"])
        
        with eda_tab1:
            st.markdown("### Word Cloud Visualization")
            st.markdown("*Most frequent words in restaurant reviews*")
            
            # Check if wordcloud image exists
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
                📁 **Word Cloud image not found.**
                
                Please upload one of these files:
                - `images/wordcloud.png`
                - `figures/wordcloud.png`
                
                The word cloud shows the most frequently mentioned words in reviews,
                with larger words indicating higher frequency.
                """)
                
                # Show placeholder description
                st.markdown("""
                **Expected insights from Word Cloud:**
                - Dominant words: *food, service, good, delicious, staff, place*
                - Cuisine mentions: *chicken, rice, noodle, steak*
                - Experience words: *amazing, excellent, recommend, return*
                """)
        
        with eda_tab2:
            st.markdown("### N-gram Analysis")
            st.markdown("*Most common word combinations in reviews*")
            
            # Check if ngram image exists
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
                📁 **N-gram image not found.**
                
                Please upload one of these files:
                - `images/ngram.png`
                - `figures/ngram_analysis.png`
                
                N-gram analysis shows common 2-word and 3-word phrases.
                """)
                
                # Show placeholder description
                st.markdown("""
                **Expected insights from N-gram Analysis:**
                
                **Bigrams (2 words):**
                - *food good*, *service excellent*, *highly recommend*
                - *nasi lemak*, *fried rice*, *ice cream*
                
                **Trigrams (3 words):**
                - *highly recommend place*, *will come back*, *food service good*
                - *value for money*, *friendly staff helpful*
                """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.7); padding: 1rem;">
    <p><strong>KL Dining Assistant</strong> — A Topic-Based Restaurant Recommendation Platform</p>
    <p>Using LDA Topic Modeling & RoBERTa Sentiment Analysis</p>
    <p>Master of Data Science Thesis Project | University of Malaya | 2025</p>
    <p>By: Tanisya Pristi Azrelia (24088031)</p>
</div>
""", unsafe_allow_html=True)
