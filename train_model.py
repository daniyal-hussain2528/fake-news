import pandas as pd
import numpy as np
import os
import re
import string
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

# Create folders
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('graphs', exist_ok=True)

# ---- TEXT PREPROCESSING ----
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# ---- LOAD DATASET ----
def load_data():
    # Look for the WELFake dataset or any dataset in the data folder
    dataset_path = None
    possible_files = [
        'data/train.csv',
        'data/WELFake_Dataset.csv',
        'data/fake_news.csv',
        'data/news.csv',
        'data/dataset.csv'
    ]

    for file in possible_files:
        if os.path.exists(file):
            dataset_path = file
            break

    if dataset_path:
        print(f"Dataset found: {dataset_path}")
        df = pd.read_csv(dataset_path)

        # Clean column names (lowercase, remove spaces)
        df.columns = df.columns.str.strip().str.lower()

        # Remove unnamed index column if present
        df = df.loc[:, ~df.columns.str.contains('^unnamed')]

        # We need at least 'text' and 'label'
        if 'text' not in df.columns or 'label' not in df.columns:
            print("Dataset does not have required 'text' and 'label' columns.")
            print("Available columns:", list(df.columns))
            print("Creating sample dataset instead...")
            return create_sample_data()

        # Combine title + text if title exists (better accuracy)
        if 'title' in df.columns:
            df['text'] = df['title'].astype(str) + " " + df['text'].astype(str)

        # Keep only needed columns
        df = df[['text', 'label']].copy()

        # Convert label to numeric
        df['label'] = pd.to_numeric(df['label'], errors='coerce')

        # Drop missing values
        df = df.dropna(subset=['text', 'label'])

        # Make sure label is 0 or 1
        df = df[df['label'].isin([0, 1])]

        print(f"Dataset loaded: {df.shape}")
        return df

    # No dataset found, create sample
    print("No real dataset found. Creating sample dataset...")
    return create_sample_data()

# ---- CREATE SAMPLE DATASET (fallback) ----
def create_sample_data():
    np.random.seed(42)

    real_news = [
        "The government announced new economic policies today.",
        "Scientists discover new species in Amazon rainforest.",
        "Local team wins national championship after hard match.",
        "New hospital opens in downtown area to serve community.",
        "Stock market reaches record high amid positive earnings.",
        "Researchers develop vaccine for tropical disease.",
        "City council approves funding for public schools.",
        "International summit addresses climate change concerns.",
        "Technology company releases new smartphone model.",
        "Agricultural exports increase for third consecutive quarter."
    ]

    fake_news = [
        "Aliens landed in New York and took control of government.",
        "Scientists prove that the moon is made of cheese.",
        "Secret society controls world economy from underground base.",
        "Celebrity found to be robot after years of pretending human.",
        "Drinking soda makes you immune to all diseases.",
        "World leaders are actually lizard people in disguise.",
        "Ancient prophecy predicts end of world next Tuesday.",
        "Magic pill discovered that lets you live forever.",
        "Underground city found beneath major metropolitan area.",
        "Time traveler shares lottery numbers from future."
    ]

    texts = []
    labels = []
    for i in range(500):
        texts.append(real_news[i % len(real_news)])
        labels.append(1)  # Real
        texts.append(fake_news[i % len(fake_news)])
        labels.append(0)  # Fake

    df = pd.DataFrame({'text': texts, 'label': labels})
    df.to_csv('data/sample_fake_news.csv', index=False)
    print(f"Sample dataset created: {df.shape}")
    return df

# ---- TRAIN MODELS ----
def train_models():
    df = load_data()

    # Preprocessing
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df.dropna(subset=['cleaned_text'])

    # Features and target
    X = df['cleaned_text']
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF Vectorizer
    # max_features reduced to 3000 for better memory on large datasets
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Save vectorizer
    pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

    # Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            max_depth=20,  # limit depth to save memory
            n_jobs=-1
        ),
        'SVM': LinearSVC(random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm
        }

        # Save model
        pickle.dump(model, open(f"models/{name.replace(' ', '_').lower()}_model.pkl", 'wb'))

        # Save confusion matrix plot
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f"graphs/{name.replace(' ', '_').lower()}_cm.png")
        plt.close()

        print(f"{name} - Accuracy: {acc:.4f}")

    # Save results
    pickle.dump(results, open('models/results.pkl', 'wb'))

    # Save comparison chart
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [v['accuracy'] for v in results.values()],
        'Precision': [v['precision'] for v in results.values()],
        'Recall': [v['recall'] for v in results.values()],
        'F1-Score': [v['f1'] for v in results.values()]
    })

    plt.figure(figsize=(10, 6))
    metrics_df.plot(x='Model', kind='bar', rot=45, width=0.8)
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('graphs/model_comparison.png')
    plt.close()

    print("Training complete! All models and graphs saved.")
    return results

if __name__ == '__main__':
    train_models()