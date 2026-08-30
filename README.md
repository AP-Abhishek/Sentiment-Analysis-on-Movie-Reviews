# Sentiment Analysis on Movie Reviews

A Data Science and Natural Language Processing (NLP) project built using Python, Scikit-Learn, and Jupyter Notebook to preprocess, analyze, and classify movie reviews into positive or negative sentiments.

## Project Overview & Notebook Workflow

The core analysis and experimentation for this project are conducted in [Sentiment_Analysis.ipynb](https://github.com/AP-Abhishek/Sentiment-Analysis-on-Movie-Reviews/blob/main/Sentiment_Analysis.ipynb). The notebook follows a complete end-to-end Data Science workflow:

1. **Dataset Loading:** Downloads and loads the IMDB Movie Reviews dataset.
2. **Data Preprocessing:** Cleans raw review texts by stripping HTML break tags (`<br />`) and converting text to lowercase.
3. **Feature Extraction:** Converts text into ~452,985 numerical feature vectors using **TF-IDF Vectorization** with unigram and bigram ranges (`ngram_range=(1, 2)`, `min_df=3`).
4. **Data Splitting:** Splits dataset into 80% training set and 20% testing set (`random_state=42`).
5. **Model Building & Training:** Trains three classification algorithms:
   * **Linear Support Vector Classifier (LinearSVC)**
   * **Logistic Regression**
   * **Multinomial Naive Bayes**
6. **Model Evaluation & Visualization:** Evaluates predictions using Accuracy, Precision, Recall, and F1-Score, complete with grouped bar charts rendered via `matplotlib`.

---

## Dataset Details
* **Name:** Kaggle - IMDB Dataset (`endofnight17j03/imdb-movies-review`)
* **File Size:** ~66.2 MB (`IMDB Dataset.csv`)
* **Total Entries:** 50,000 movie reviews (balanced positive/negative distribution)
* **Features:**
  * `review`: Raw text string of the movie review.
  * `sentiment`: Target binary label (`positive` / `negative`).

```python
import kagglehub

path = kagglehub.dataset_download("endofnight17j03/imdb-movies-review")
```

---

## Model Performance & Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear SVM (LinearSVC)** | **0.92** | **0.91** | **0.93** | **0.92** |
| **Logistic Regression** | 0.91 | 0.90 | 0.92 | 0.91 |
| **Multinomial Naive Bayes** | 0.89 | 0.91 | 0.87 | 0.89 |

*Linear Support Vector Classifier achieved the highest performance with **92.08% Accuracy** and **0.92 F1-Score**.*

---

## Environment Setup & Running the Notebook

To run the primary Jupyter Notebook on your local machine:

### Prerequisites
* Python 3.8 or higher installed on your system.
* Git installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/AP-Abhishek/Sentiment-Analysis-on-Movie-Reviews.git
cd Sentiment-Analysis-on-Movie-Reviews
```

### Step 2: Create a Virtual Environment
* **On Windows (Command Prompt / PowerShell):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter Notebook
```bash
jupyter notebook Sentiment_Analysis.ipynb
```
Once open, click **Cell > Run All** (or **Kernel > Restart & Run All**) to execute data preprocessing, TF-IDF vectorization, training, and graph generation.

---

## Project Extension: Command Line Interface (CLI)

In addition to the notebook exploration, the pipeline has been packaged into modular Python scripts for CLI usage:

### 1. Train & Export Model Artifacts
```bash
# Train default LinearSVC model
python main.py train

# Or specify model: svm, lr (Logistic Regression), or nb (Naive Bayes)
python main.py train -m lr
```

### 2. Predict Sentiments via Terminal

#### Direct Command Argument:
```bash
python main.py predict --review "The movie was fantastic with brilliant acting!"
```

#### Interactive Terminal Session:
```bash
python main.py -i
```

---

## Project Structure

```text
Sentiment Analysis on Movie Reviews/
├── dataset/
│   └── IMDB Dataset.csv            # IMDB Dataset (~66.2 MB, ignored in git)
├── Sentiment_Analysis.ipynb        # Primary Jupyter Notebook (EDA, Models & Graphs)
├── models/                         # Saved model artifacts (.pkl, ignored in git)
├── src/
│   ├── __init__.py                 # Package version definition
│   ├── train.py                    # Modular training script
│   └── predict.py                  # Prediction engine & interactive session
├── main.py                         # CLI entry point
├── pyproject.toml                  # Python package configuration
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```
