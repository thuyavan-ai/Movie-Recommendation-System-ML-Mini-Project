# 🎬 CineMatch — Movie Recommendation System

A beginner-friendly machine-learning mini project that recommends movies similar to a movie selected by the user. It uses **content-based filtering**, **TF-IDF text vectorization**, and **cosine similarity**, with a Streamlit web interface.

## Features

- Select a movie from a dropdown.
- Get 3–10 similar movie recommendations.
- View genres, descriptions, and similarity scores.
- Browse the included sample dataset.
- Runs locally; no API key or account is needed.

## Tech stack

- Python
- Pandas
- scikit-learn
- Streamlit

## Project structure

```text
movie-recommendation-system/
├── app.py
├── recommender.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── movies.csv
```

## Run locally

### 1. Install Python

Use Python 3.10 or newer (Python 3.11 or 3.12 is a good choice for broad package compatibility).

### 2. Open a terminal in this project folder

Create and activate a virtual environment.

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the app
```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

## How it works

1. The app loads movie titles, genres, and descriptions from `data/movies.csv`.
2. Genre and description text is combined. Genres are repeated to give them extra weight.
3. `TfidfVectorizer` converts the text into numerical vectors.
4. Cosine similarity compares each movie's vector with the selected movie.
5. The system returns the most similar titles, excluding the selected movie.

## Dataset

This repository includes a small **sample dataset** for demonstration. It is intended for learning and prototype presentations, not as a complete movie catalog. You can replace or expand `data/movies.csv` with a dataset you have permission to use, keeping the columns `title`, `genres`, and `description`.

For a larger public movie dataset, see the [MovieLens datasets](https://grouplens.org/datasets/movielens/) and follow its usage terms. Descriptions may need to be added separately.

## Limitations

- The sample dataset is small, so recommendations are limited to the movies included.
- This is content-based filtering, not personalized collaborative filtering from many users' ratings.
- Similarity scores measure text similarity; they are not probabilities or quality ratings.
- Movie descriptions in the included dataset are short demonstration summaries.

## Ideas for future improvements

- Add posters using a properly licensed movie metadata API.
- Add genre and rating filters.
- Add user profiles and watch history.
- Evaluate recommendations with a larger dataset.
- Build a hybrid model combining content features and user ratings.

## GitHub upload

1. Create a new repository on GitHub, for example `movie-recommendation-system`.
2. Extract this ZIP file.
3. Open a terminal in the extracted folder.
4. Run:

```bash
git init
git add .
git commit -m "Add movie recommendation system mini project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/movie-recommendation-system.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Create the empty repository on GitHub first; do not add a README there if you already have this README locally.

## License

This project is provided for educational use. If you plan to publish or redistribute a larger third-party dataset, check that dataset's license and terms first.
