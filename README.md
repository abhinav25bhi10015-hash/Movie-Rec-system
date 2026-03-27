# MovieMind AI: Hybrid Movie Recommendation System

A sophisticated Machine Learning application built with **Python** and **Streamlit** that provides personalized movie suggestions by combining **Content-Based Filtering** with **Statistical Popularity** metrics.

---

## 📖 Overview
This project solves the "Information Overload" problem in digital libraries. Unlike basic recommenders that only look at genres, **Cinematch AI** utilizes a hybrid engine to ensure recommendations are both thematically relevant and highly rated by the community.

## 🧠 How it Works (The AI Logic)
The system operates using a three-stage pipeline:

1.  **Vectorization (NLP):** Using `TfidfVectorizer`, the system converts movie genres into a high-dimensional matrix. It calculates the **TF-IDF** (Term Frequency-Inverse Document Frequency) score to down-weight common genres and highlight unique ones.
2.  **Similarity Math:** We calculate the **Cosine Similarity** between movie vectors. This measures the cosine of the angle between two vectors, determining how "close" two movies are in a geometric space.
3.  **Hybrid Filtering:** The "Smart" layer merges the content results with a **Ratings Dataset**. It applies a threshold filter (minimum 15 user ratings) and sorts by average rating to ensure only quality movies are suggested.

---

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **ML Libraries:** Scikit-Learn, Pandas, Numpy
* **UI Framework:** Streamlit
* **Search Logic:** FuzzyWuzzy (Levenshtein Distance for typo-tolerance)

---

## 📁 Project Structure
```plaintext
movie_rec/
├── app.py                  # Core ML Logic & Testing script
├── web_app.py              # Streamlit Web Interface (Main Entry)
├── movies.csv              # Dataset: Movie metadata
├── ratings.csv             # Dataset: User ratings
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup
To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/movie-recommender.git
    cd movie-recommender
    ```

2.  **Set up Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install pandas scikit-learn streamlit fuzzywuzzy python-Levenshtein
    ```

4.  **Run the Application:**
    ```bash
    streamlit run web_app.py
    ```

---

## 📊 Dataset Attribution
This project uses the **MovieLens Latest Small Dataset** provided by GroupLens Research.
* **Source:** [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
* **Size:** ~100,000 ratings across 9,000 movies.

---

## 📈 Future Improvements
* **Deep Learning:** Implementing Neural Collaborative Filtering (NCF).
* **API Integration:** Fetching real-time posters from the TMDB API.
* **Cloud Deployment:** Hosting the app on Streamlit Cloud or AWS.
