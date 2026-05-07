import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

print("Loading CSVs...")
books = pd.read_csv('Books.csv', low_memory=False)
users = pd.read_csv('Users.csv', low_memory=False)
ratings = pd.read_csv('Ratings.csv', low_memory=False)

# ── Clean column names / types ──────────────────────────────────────────────
books.columns   = books.columns.str.strip()
users.columns   = users.columns.str.strip()
ratings.columns = ratings.columns.str.strip()

ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce')

# ── Popular books ────────────────────────────────────────────────────────────
print("Building popular_df...")

ratings_with_name = ratings.merge(books, on='ISBN')

num_rating_df = (ratings_with_name.groupby('Book-Title')
                 .count()['Book-Rating']
                 .reset_index()
                 .rename(columns={'Book-Rating': 'num_ratings'}))

avg_rating_df = (ratings_with_name.groupby('Book-Title')
                 ['Book-Rating']
                 .mean()
                 .reset_index()
                 .rename(columns={'Book-Rating': 'avg_rating'}))

popular_df = (num_rating_df.merge(avg_rating_df, on='Book-Title')
              .merge(books, on='Book-Title')
              .drop_duplicates('Book-Title')
              [['Book-Title', 'Book-Author', 'Image-URL-M',
                'num_ratings', 'avg_rating']])

popular_df = (popular_df[popular_df['num_ratings'] >= 250]
              .sort_values('avg_rating', ascending=False)
              .head(50))

pickle.dump(popular_df, open('popular.pkl', 'wb'))
print(f"  popular.pkl saved  ({len(popular_df)} books)")

# ── Collaborative filtering ──────────────────────────────────────────────────
print("Building pt & similarity_scores...")

x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
power_users = x[x].index

filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(power_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

pt = final_ratings.pivot_table(
    index='Book-Title',
    columns='User-ID',
    values='Book-Rating'
).fillna(0)

similarity_scores = cosine_similarity(pt)

pickle.dump(pt,               open('pt.pkl',               'wb'))
pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'))
print(f"  pt.pkl saved  ({pt.shape})")
print(f"  similarity_scores.pkl saved  ({similarity_scores.shape})")

# ── Books df ─────────────────────────────────────────────────────────────────
pickle.dump(books, open('books.pkl', 'wb'))
print(f"  books.pkl saved  ({len(books)} rows)")

print("\nDone! All .pkl files regenerated successfully.")
