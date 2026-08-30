import os
import re
import sys
import joblib

def load_artifacts():
    model_path = os.path.join("models", "model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Saved model artifacts not found in 'models/' folder.")
        print("Please run 'python main.py train' first.")
        sys.exit(1)
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_sentiment(review_text: str) -> str:
    model, vectorizer = load_artifacts()
    clean_text = re.sub(r"<br />", " ", review_text).lower()
    transformed = vectorizer.transform([clean_text])
    prediction = model.predict(transformed)[0]
    return "Positive" if prediction == 1 else "Negative"

def interactive_mode():
    model, vectorizer = load_artifacts()
    print("================================================")
    print("\tMovie Review Sentiment Predictor")
    print("================================================")
    print("Type a movie review below (type 'exit' to quit):\n")

    while True:
        try:
            user_input = input("Enter Review > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            clean_text = re.sub(r"<br />", " ", user_input).lower()
            transformed = vectorizer.transform([clean_text])
            prediction = model.predict(transformed)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"
            print(f"Predicted Sentiment: {sentiment}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break