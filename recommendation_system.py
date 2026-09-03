import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('imdb_top_250.csv')

df['genres'] = df['genres'].fillna('')
df['description'] = df['description'].fillna('')

df['combined_features'] = (df['genres'] + ' ') * 3 + df['description']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

similarity_matrix = cosine_similarity(tfidf_matrix)

print(tfidf_matrix.shape)
print(similarity_matrix.shape)

def recommend(movie_title, num_recommendations=5):
    matches = df[df['primaryTitle'].str.lower() == movie_title.lower()]
    
    if matches.empty:
        print("Movie not found in the dataset.")
        return

    idx = matches.index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:num_recommendations + 1]

    print(f"Movies similar to '{df.iloc[idx]['primaryTitle']}':")
    for i, score in scores:
        print(f"{df.iloc[i]['primaryTitle']} (similarity: {score:.2f})")

recommend('back to the future')
