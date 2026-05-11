# 📚 Book Recommender System

A machine learning web app that recommends books based on what you've read. Built with Flask and trained on the Book-Crossing Dataset with over **1 million ratings**.

---

## What it does

- **Home page** — shows the Top 50 popular books, ranked by average rating and number of votes.
- **Recommend page** — type any book title and get 4 similar books instantly using collaborative filtering.

The recommendation engine filters active users (200+ ratings) and well-rated books (50+ reviews), builds a user-item matrix, and uses **Cosine Similarity** to find the closest matches. It also supports partial, case-insensitive title search so you don't need to type the exact name.

---

## Tech Stack

- **Backend** — Python, Flask
- **ML** — Scikit-learn (Cosine Similarity), Pandas, NumPy
- **Frontend** — HTML, Bootstrap, Jinja2
- **Deployment** — Gunicorn (Heroku / Render ready)

---

## Getting Started

```bash
git clone https://github.com/khushi-solanki09/Book-Recommender-System.git
cd Book-Recommender-System
pip install -r requirements.txt

# Optional: regenerate model files from raw CSVs
python regenerate_pkl.py

# Run the app
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## Dataset

Uses the [Book-Crossing Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) — ~271K books, ~278K users, and ~1.1M ratings.

---

## What I'd add next

- User login + personal recommendation history
- Content-based filtering (genre, description)
- REST API for mobile clients

---

Made by **Khushi Solanki** · [GitHub](https://github.com/khushi-solanki09)
