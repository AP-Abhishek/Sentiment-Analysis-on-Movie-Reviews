import os
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

def train_model(model_type: str = "svm"):
    model_type = model_type.lower()
    if model_type == "lr":
        model_name = "Logistic Regression"
        model = LogisticRegression(max_iter=1000)
    elif model_type == "nb":
        model_name = "Multinomial Naive Bayes"
        model = MultinomialNB()
    else:
        model_name = "Linear Support Vector Classifier (LinearSVC)"
        model = LinearSVC()

    print(f"\nTraining Model: {model_name}\n")

    dataset_path = os.path.join("dataset", "IMDB Dataset.csv")
    models_dir = "models"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at '{dataset_path}'.")
        print("Please place 'IMDB Dataset.csv' inside the 'dataset/' folder.")
        return
    
    os.makedirs(models_dir, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)

    print("Preprocessing text data...")
    df['review'] = df['review'].astype(str).str.replace("<br />", " ", regex=False).str.lower()

    print("Extracting TF-IDF feature...")
    vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment'].map({'positive': 1, 'negative': 0})

    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Fitting {model_name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nTraining Complete!")
    print(f"Test Accuracy: {acc * 100:.2f}% | Test F1-Score: {f1:.2f}\n")

    model_path = os.path.join(models_dir, "model.pkl")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

    print(f"Saving artifacts to '{models_dir}/...'")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Saved 'model.pkl' and 'tfidf_vectorizer.pkl' successfully!")
