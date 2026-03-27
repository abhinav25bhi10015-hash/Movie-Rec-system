# MovieMind AI: Hybrid Movie Recommendation System

This is a smart application that uses Machine Learning to suggest movies to people. It is built with **Python** and **Streamlit**. The application provides movie suggestions by combining what the movies are about with how popular they're

---

## Overview

This project helps solve the problem of having many movies to choose from. Unlike recommenders that only look at what kind of movie it is, **Cinematch AI** uses a special engine to make sure the movie suggestions are both about something similar and liked by a lot of people.

## How it Works

The system works in three steps:

1.  **Vectorization**: The system takes the kinds of movies. Turns them into a special set of numbers. It calculates a score to make sure common kinds of movies do not get much attention and unique kinds of movies get noticed.

2.  **Similarity Math**: We calculate how similar two movies are by looking at the numbers. This measures how "close" two movies are.

3.  **Hybrid Filtering**: The special layer combines what the movies are about with what people think of them. It only suggests movies that a lot of people have rated and that have an average rating.

---

## Tech Stack

* **Language**: Python 3.11+

* **ML Libraries**: Scikit-Learn, Pandas, Numpy

* **UI Framework**: Streamlit

* **Search Logic**: FuzzyWuzzy

---

## Project Structure

```plaintext

movie_rec/

├── app.py                  # Core ML Logic & Testing script

├── web_app.py              # Streamlit Web Interface (Main Entry)

├── movies.csv              # Dataset: Movie metadata

├── ratings.csv             # Dataset: User ratings

└── README.md               # Project documentation

```

---

## Installation & Setup

To run this project on your computer follow these steps:

1.  **Get the project code**:

```bash

git clone https://github.com/your-username/movie-recommender.git

cd movie-recommender

```

2.  **Set up an environment**:

```bash

python3 -m venv venv

source venv/bin/activate

```

3.  **Get the tools**:

```bash

pip install pandas scikit-learn streamlit fuzzywuzzy python-Levenshtein

```

4.  **Run the application**:

```bash

streamlit run web_app.py

```

---

## Dataset Attribution

This project uses the **MovieLens Latest Small Dataset** from GroupLens Research.

* **Source**: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

* **Size**: ~100,000 ratings across 9,000 movies.

---

## Future Improvements

* **Deep Learning**: Use Neural Collaborative Filtering to make the suggestions even better.

* **API Integration**: Get the movie posters from the TMDB API.

* **Cloud Deployment**: Host the application, on Streamlit Cloud or AWS.
