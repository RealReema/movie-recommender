import streamlit as st
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

st.title("Movie Recommender")
st.write("Pick a movie and get 5 similar recommendations from IMDb's Top 250 list.")

movie_list = sorted(df['primaryTitle'].tolist())
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    idx = df[df['primaryTitle'] == selected_movie].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:6]

    st.subheader(f"Movies similar to {selected_movie}:")
    for i, score in scores:
        st.write(f"**{df.iloc[i]['primaryTitle']}** - similarity: {score:.2f}")
