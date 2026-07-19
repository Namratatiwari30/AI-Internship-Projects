import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Create a clean movie database matrix
movies_data = {
    'Title': [
        'The Dark Knight', 'Inception', 'Interstellar', 
        'The Hangover', 'Superbad', 'The Conjuring', 
        'Insidious', 'Avatar'
    ],
    'Genre': [
        'Action Crime Drama Thriller', 'Action Sci-Fi Thriller', 'Adventure Drama Sci-Fi',
        'Comedy', 'Comedy', 'Horror Mystery Thriller',
        'Horror Mystery Thriller', 'Action Adventure Fantasy Sci-Fi'
    ]
}

df = pd.DataFrame(movies_data)

# 2. Convert text genres into numerical vectors using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Genre'])

# 3. Compute the Cosine Similarity scores between all movies
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. Recommendation Function
def get_recommendations(movie_title):
    if movie_title not in df['Title'].values:
        return f"Movie '{movie_title}' not found."
        
    idx = df[df['Title'] == movie_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top similar movies
    movie_indices = [i[0] for i in sim_scores[1:3]]
    return df['Title'].iloc[movie_indices].tolist()

# 5. Test the Recommendation Engine
target_movie = "Inception"
recommendations = get_recommendations(target_movie)

print("\n--- AI Movie Recommendation System ---")
print(f"Because you watched: '{target_movie}'")
print(f"Recommended Movies: {recommendations}\n")
