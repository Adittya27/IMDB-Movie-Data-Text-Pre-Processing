# IMDB Text Preprocessing
Link for the Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews.



A complete text preprocessing pipeline for the [IMDB Movie Reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (50,000 reviews), built in Python for use in Google Colab. This notebook takes raw, noisy review text and turns it into clean, model-ready features using classic NLP techniques.

## What it does

- **Data loading & exploration** — loads `IMDB Dataset.csv`, checks shape, missing values, and class distribution (positive/negative sentiment).
- **Text cleaning pipeline**
  - Lowercasing
  - HTML tag removal (`BeautifulSoup`)
  - URL removal
  - Contraction expansion (`don't` → `do not`)
  - Punctuation & number removal
  - Extra whitespace cleanup
- **Tokenization & normalization**
  - Word tokenization (`nltk`)
  - Stopword removal
  - Lemmatization (WordNet)
- **Reusable pipeline** — all steps combined into a single `clean_text()` function with configurable options (lemmatization vs. stemming, stopword removal, minimum token length).
- **Visualization**
  - Top 20 most frequent words (bar chart)
  - Word cloud of cleaned text
  - Raw vs. cleaned word count distribution
- **Feature extraction**
  - Bag-of-Words (`CountVectorizer`)
  - TF-IDF (`TfidfVectorizer`)
  - Side-by-side comparison of BoW sums vs. TF-IDF means for top terms
- **Export** — saves the cleaned dataset to CSV for downstream modeling.

## Requirements

```bash
pip install kaggle nltk beautifulsoup4 wordcloud contractions scikit-learn
```

## Usage

1. Open the notebook in Google Colab (or Jupyter).
2. Place `IMDB Dataset.csv` in the working directory (`/content/` in Colab.
3. Run all cells to clean the text, visualize results, and generate BoW/TF-IDF feature matrices.

## Output

- `cleaned_dataset.csv` — the cleaned review text, ready for downstream sentiment analysis or ML modeling.
