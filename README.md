# Sentiment Analysis of Movies

A project in which the movies dataset with reviews, has been used to train few models to predict future sentiments of movie reviews.

### About Dataset
Name: Kaggle - IMDB Dataset

Entries: 50,000

Features: Review and Sentiment
```
kagglehub.dataset_download("endofnight17j03/imdb-movies-review")
```

### Steps
1. Downloading from Kaggle
1. Data Pre-processing
1. Feature Extraction
1. Data Splitting
1. Model Building & Training
1. Model Evaluation

### Models
Models used for the analysis are:
1. Logistic Regression Model
1. Multinomial Naive Bayes Model
1. SVM (Linear)

### Model Evaluation Metrics
1. Accuracy
1. Precision
1. Recall
1. F1-score

### Project Structure
```
Sentiment Analysis on Movie Reviews/
-> dataset/
    -> IMDB Dataset.csv
-> Sentiment_Analysis.ipynb
-> README.md
```