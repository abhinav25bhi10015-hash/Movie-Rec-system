import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load BOTH Datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# 2. Pre-process Genres
movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)

# 3. ML Logic: TF-IDF & Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. Smart Recommendation Function
def get_smart_recommendations(movie_title):
    try:
        # Find the movie index
        idx = movies[movies['title'] == movie_title].index[0]
        
        # Get top 50 similar movies based on Genre (Content)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:50]
        movie_indices = [i[0] for i in sim_scores]
        
        # Create a candidate DataFrame
        candidates = movies.iloc[movie_indices].copy()
        
        # --- THE SMART PART: Calculate Popularity Stats ---
        movie_stats = ratings.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            vote_count=('rating', 'count')
        ).reset_index()
        
        # Merge stats with our candidates
        candidates = candidates.merge(movie_stats, on='movieId', how='left')
        
        # Filter: Must have at least 20 ratings (avoids obscure "bad" movies)
        # Sort by average rating (shows the BEST similar movies)
        smart_recs = candidates[candidates['vote_count'] > 20].sort_values(by='avg_rating', ascending=False)
        
        return smart_recs[['title', 'avg_rating']].head(5)
    
    except IndexError:
        return "Movie not found. Please check the spelling/year!"

