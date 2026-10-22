import streamlit as st
import pandas as pd
from recommender import MovieRecommender

st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")

@st.cache_resource
def load_recommender():
    return MovieRecommender("data/movies.csv")

recommender = load_recommender()

st.title("🎬 CineMatch")
st.subheader("Movie Recommendation System")
st.write("Discover movies based on genres and story keywords using content-based filtering.")

with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        **Method:** Content-based filtering  
        **Features:** TF-IDF + cosine similarity  
        **Dataset:** Included sample movie dataset  
        **Stack:** Python, pandas, scikit-learn, Streamlit
        """
    )
    st.caption("Educational mini project. Recommendations depend on the included dataset.")

movies = recommender.movie_titles
if not movies:
    st.error("No movies found. Check that data/movies.csv exists and contains a title column.")
    st.stop()

selected = st.selectbox("Choose a movie you like", movies, index=0)
count = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
if st.button("Recommend movies", type="primary"):
    results = recommender.recommend(selected, n=count)
    st.markdown(f"### Because you selected **{selected}**")
    if results.empty:
        st.info("No recommendations found for this title.")
    else:
        cols = st.columns(min(5, len(results)))
        for idx, (_, movie) in enumerate(results.iterrows()):
            with cols[idx % len(cols)]:
                st.markdown(f"**{movie['title']}**")
                st.caption(movie["genres"] if pd.notna(movie["genres"]) else "Genre not listed")
                st.write(f"Similarity: {movie['similarity']:.0%}")
                description = movie.get("description", "")
                if pd.notna(description) and str(description).strip():
                    with st.expander("Movie details"):
                        st.write(description)
else:
    st.info("Select a movie, choose how many suggestions you want, and click **Recommend movies**.")

with st.expander("Browse the dataset"):
    st.dataframe(recommender.movies[["title", "genres", "description"]], use_container_width=True, hide_index=True)

st.divider()
st.caption("Built as an educational machine-learning mini project.")
