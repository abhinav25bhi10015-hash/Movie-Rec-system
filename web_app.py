import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

st.set_page_config(page_title="MovieMind AI", layout="wide")
st.title("🧠 MovieMind AI")


@st.cache_data
def load_all_data():
    m = pd.read_csv('movies.csv')
    r = pd.read_csv('ratings.csv')
    m['genres_clean'] = m['genres'].str.replace('|', ' ', regex=False)
    
    # Pre-calculate global movie stats
    stats = r.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        vote_count=('rating', 'count')
    ).reset_index()
    
    return m, r, stats

movies, ratings, movie_stats = load_all_data()

# 3. Setup ML Model
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. The Smart Logic (Now integrated into the UI)
def get_hybrid_recs(movie_title):
    # Find closest match to handle typos
    all_titles = movies['title'].tolist()
    closest_match, score = process.extractOne(movie_title, all_titles)
    
    if score < 60:
        return None, "Movie not found. Try another name!"
    
    idx = movies[movies['title'] == closest_match].index[0]
    
    # Content-based similarity (Top 40)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:40]
    movie_indices = [i[0] for i in sim_scores]
    
    # Filter by Popularity & Quality
    candidates = movies.iloc[movie_indices].copy()
    candidates = candidates.merge(movie_stats, on='movieId', how='left')
    
    # Must have at least 15 ratings, sort by the highest average
    final_recs = candidates[candidates['vote_count'] > 15].sort_values(by='avg_rating', ascending=False)
    
    return closest_match, final_recs.head(6)

# 5. The Sidebar Interface
st.sidebar.header("User Input")
search_query = st.sidebar.text_input("Type a movie you like:", "Toy Story")

if st.sidebar.button("Find Recommendations"):
    actual_title, results = get_hybrid_recs(search_query)
    
    if isinstance(results, str):
        st.error(results)
    else:
        st.success(f"Showing results based on: **{actual_title}**")
        st.markdown("---")
        
        # Display in a clean Grid
        cols = st.columns(3)
        for i, row in results.reset_index().iterrows():
            with cols[i % 3]:
                st.info(f"**{row['title']}**")
                # Visual star rating
                stars = "⭐" * int(round(row['avg_rating']))
                st.write(f"Rating: {stars} ({row['avg_rating']:.1f}/5)")
                st.caption(f"Genres: {row['genres']}")
else:
    st.info("👈 Enter a movie name in the sidebar and click 'Find'!")