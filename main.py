import argparse
from src import __version__
from src.train import train_model
from src.predict import predict_sentiment, interactive_mode

def main():
    parser = argparse.ArgumentParser(
        prog="sentiment-analyzer-cli",
        description="Movie Review Sentiment Analysis CLI"
    )
    
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "-i", 
        "--interactive",
        action="store_true",
        help="Launch interactive prediction mode directly"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        help="Available Commands"
    )
    
    train_parser = subparsers.add_parser(
        "train", 
        help="Train model & export artifacts to models/",
        description="Train a classification model (Linear SVM, Logistic Regression, or Naive Bayes) on the IMDB dataset and export trained artifacts to 'models/'."
    )

    train_parser.add_argument(
        "-m", 
        "--model",
        choices=["svm", "lr", "nb"],
        default="svm",
        metavar="MODEL",
        help="Algorithm to train: svm (default), lr, nb"
    )

    predict_parser = subparsers.add_parser(
        "predict", 
        help="Predict sentiment for a review",
        description="Analyze and classify movie reviews into Positive or Negative sentiments."
    )
    
    predict_parser.add_argument(
        "-r",
        "--review",
        type=str,
        help="Review text to classify"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.command == "train":
        train_model(model_type=args.model)
    elif args.command == "predict":
        if args.review:
            result = predict_sentiment(args.review)
            print(f"\nReview: \"{args.review}\"")
            print(f"Predicted Sentiment: {result}\n")
        else:
            interactive_mode()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
