from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """Content-based movie recommender using TF-IDF and cosine similarity."""

    def __init__(self, csv_path="data/movies.csv"):
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}. Run the app from the project root."
            )

        self.movies = pd.read_csv(path).fillna("")
        required = {"title", "genres", "description"}
        missing = required.difference(self.movies.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

        self.movies["title"] = self.movies["title"].astype(str)
        self.movies["genres"] = self.movies["genres"].astype(str)
        self.movies["description"] = self.movies["description"].astype(str)
        self.movies = self.movies.drop_duplicates(subset=["title"]).reset_index(drop=True)

        # Give genres extra weight by including them twice in the text.
        self.movies["combined_features"] = (
            self.movies["genres"] + " " + self.movies["genres"] + " " +
            self.movies["description"]
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.feature_matrix = self.vectorizer.fit_transform(
            self.movies["combined_features"]
        )
        self.similarity_matrix = cosine_similarity(self.feature_matrix)

        self.movie_titles = self.movies["title"].tolist()
        self._title_to_index = {
            title.casefold(): index for index, title in enumerate(self.movie_titles)
        }

    def recommend(self, title, n=5):
        """Return up to n similar movies, excluding the selected title."""
        key = str(title).casefold()
        if key not in self._title_to_index:
            raise ValueError(f"Movie not found: {title}")

        index = self._title_to_index[key]
        scores = list(enumerate(self.similarity_matrix[index]))
        scores.sort(key=lambda item: item[1], reverse=True)
        scores = [
            (movie_index, score)
            for movie_index, score in scores
            if movie_index != index
        ][:max(0, int(n))]

        rows = []
        for movie_index, score in scores:
            movie = self.movies.iloc[movie_index]
            rows.append({
                "title": movie["title"],
                "genres": movie["genres"],
                "description": movie["description"],
                "similarity": float(score),
            })
        return pd.DataFrame(rows, columns=["title", "genres", "description", "similarity"])
