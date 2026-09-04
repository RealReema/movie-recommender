# Movie Recommender

A content-based movie recommendation system built on the IMDb Top 250 dataset. Pick a movie and get 5 similar recommendations, based on genre and plot description.

**Live app:** https://realreema-movie-recommender.streamlit.app

## Demo

https://github.com/user-attachments/assets/2f12920b-3082-4ce8-995e-c183107ab28e

## How it works

Each movie's genres and description get combined into one text feature (genres count more than the description). That text gets converted into numbers using TF-IDF, then I use cosine similarity to see how close each movie is to every other one. Give it a movie, and it returns the 5 closest matches.

## Files

- `app.py` - the Streamlit web app
- `recommendation_system.py` - the recommendation logic, tested from the terminal before building the app
- `imdb_top_250.csv` - the dataset
- `.streamlit/config.toml` - custom color theme for the app

## Tools

Python, pandas, scikit-learn, Streamlit

## Data source

The IMDb dataset is from Kaggle: [Top 250 Movies on IMDb in 2026](https://www.kaggle.com/datasets/arjunsinghgangwar/top-250-movies-on-imdb-in-2026) by Arjun Singh Gangwar (MIT License).
