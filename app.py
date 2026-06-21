import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import train_model
from theme import inject_theme, render_verdict_badge, panel_start, panel_end, COLORS

# Set page title
st.set_page_config(page_title="Fake News Identification System", layout="wide", page_icon="📰")

# Inject the 3D / energetic theme
inject_theme()

# Make matplotlib charts match the dark newsroom theme
plt.rcParams.update({
    "figure.facecolor": COLORS["bg_panel"],
    "axes.facecolor": COLORS["bg_panel"],
    "savefig.facecolor": COLORS["bg_panel"],
    "axes.edgecolor": COLORS["ink_line"],
    "axes.labelcolor": COLORS["text_primary"],
    "xtick.color": COLORS["text_dim"],
    "ytick.color": COLORS["text_dim"],
    "text.color": COLORS["text_primary"],
    "axes.titlecolor": COLORS["text_primary"],
    "grid.color": COLORS["ink_line"],
})

# Save current page in session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# ---- TOP BUTTON NAVIGATION ----
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button('🏠 Home'):
        st.session_state.page = 'Home'
with col2:
    if st.button('🤖 ML Algorithms'):
        st.session_state.page = 'ML Algorithms'
with col3:
    if st.button('📏 Model Metrics'):
        st.session_state.page = 'Model Metrics'
with col4:
    if st.button('🔧 ML Pipeline'):
        st.session_state.page = 'ML Pipeline'
with col5:
    if st.button('💻 Source Code'):
        st.session_state.page = 'Source Code'

st.title('📰 Fake News Identification System')
st.write('Detect whether a news article is **Real** or **Fake** using Machine Learning.')
st.markdown("<hr>", unsafe_allow_html=True)

# ---- HELPER FUNCTIONS ----
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_results():
    if not os.path.exists('models/results.pkl'):
        st.warning("Models not found. Training now... please wait.")
        train_model.train_models()
    return pickle.load(open('models/results.pkl', 'rb'))

def predict_news(text, model_name):
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    model_file = f"models/{model_name.replace(' ', '_').lower()}_model.pkl"
    model = pickle.load(open(model_file, 'rb'))
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    return "Real" if pred == 1 else "Fake"

def prepare_uploaded_df(df):
    # Clean column names: remove BOM, spaces, lowercase, convert to string
    df.columns = [str(col).encode('utf-8').decode('utf-8-sig').strip().lower() for col in df.columns]

    # Remove unnamed index columns
    df = df.loc[:, ~df.columns.str.contains('^unnamed')]

    # Check required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        found_cols = ', '.join(df.columns.tolist())
        return None, f'CSV must contain "text" and "label" columns. Found columns: [{found_cols}]'

    # Optional: combine title + author + text for better accuracy
    text_parts = []
    for col in ['title', 'author', 'text']:
        if col in df.columns:
            text_parts.append(df[col].astype(str))
    if text_parts:
        df['text'] = text_parts[0]
        for part in text_parts[1:]:
            df['text'] = df['text'] + " " + part

    # Keep only needed columns
    df = df[['text', 'label']].copy()

    # Convert label to numbers
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['text', 'label'])

    # Make sure no empty text
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != '']

    return df, None

# ============================================================
# HOME PAGE
# ============================================================
if st.session_state.page == 'Home':
    st.header('🏠 Home')

    panel_start("Project Brief")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.write('**Dataset Used:** Kaggle Fake News Dataset')
        st.write('**Description:** Detect misleading or real news articles.')
        st.write('**Language:** Python')
    with pcol2:
        st.write('**Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Streamlit')
        st.write('**ML Algorithms:** Logistic Regression, Naive Bayes, Random Forest, SVM')
    panel_end()

    st.subheader('📤 Upload Dataset')
    uploaded_file = st.file_uploader("Upload a CSV file (must have 'text' and 'label' columns)", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='latin-1')

        df, error_msg = prepare_uploaded_df(df)
        if error_msg:
            st.error(error_msg)
            st.stop()

        panel_start("Dataset Preview")
        st.dataframe(df.head())
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", f"{len(df)}")
        m2.metric("Real News (label=1)", f"{sum(df['label'] == 1)}")
        m3.metric("Fake News (label=0)", f"{sum(df['label'] == 0)}")
        panel_end()

        # Matplotlib bar chart — recolored to theme (Fake=alert red, Real=verified green)
        panel_start("Class Distribution")
        fig, ax = plt.subplots()
        counts = df['label'].value_counts().sort_index()
        counts.plot(kind='bar', ax=ax, color=[COLORS['alert_red'], COLORS['verified_green']])
        ax.set_xticklabels(['Fake', 'Real'], rotation=0)
        ax.set_title('Distribution of Fake vs Real News')
        ax.set_ylabel('Count')
        for spine in ax.spines.values():
            spine.set_color(COLORS['ink_line'])
        st.pyplot(fig)
        panel_end()

        # Prediction section
        st.subheader('🔍 Test a News Article')
        panel_start("Live Classifier")
        results = load_results()
        model_name = st.selectbox('Select Model', list(results.keys()))
        test_text = st.text_area('Enter news text here:', height=120)

        st.markdown('<div class="pulse-cta">', unsafe_allow_html=True)
        predict_clicked = st.button('⚡ Predict')
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_clicked:
            if test_text:
                prediction = predict_news(test_text, model_name)
                render_verdict_badge(prediction)
            else:
                st.warning('Please enter some text to predict.')
        panel_end()

    # Show accuracy graph of trained models
    st.subheader('📊 Trained Models Accuracy')
    if os.path.exists('models/results.pkl'):
        panel_start("Benchmark")
        results = load_results()
        acc_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [v['accuracy'] for v in results.values()]
        })
        st.bar_chart(acc_df.set_index('Model'), color=COLORS['violet'])
        panel_end()

# ============================================================
# ML ALGORITHMS PAGE
# ============================================================
elif st.session_state.page == 'ML Algorithms':
    st.header('🤖 Machine Learning Algorithms')

    panel_start("Models In Play")
    st.write('1. **Logistic Regression** — Good for binary classification problems.')
    st.write('2. **Naive Bayes** — Fast and effective for text data.')
    st.write('3. **Random Forest** — Uses multiple decision trees for better accuracy.')
    st.write('4. **SVM (Support Vector Machine)** — Finds the best boundary between classes.')
    panel_end()

    if os.path.exists('models/results.pkl'):
        results = load_results()
        # Comparison table
        metrics_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [v['accuracy'] for v in results.values()],
            'Precision': [v['precision'] for v in results.values()],
            'Recall': [v['recall'] for v in results.values()],
            'F1-Score': [v['f1'] for v in results.values()]
        })
        st.subheader('Model Comparison Table')
        panel_start()
        st.dataframe(metrics_df)
        panel_end()

        # Comparison chart
        st.subheader('Model Comparison Chart')
        panel_start()
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.set_index('Model').plot(
            kind='bar', ax=ax, rot=45, width=0.8,
            color=[COLORS['violet'], COLORS['verified_green'], COLORS['amber'], COLORS['alert_red']]
        )
        ax.set_ylim(0, 1.1)
        ax.set_title('Comparison of ML Algorithms')
        ax.set_ylabel('Score')
        ax.legend(loc='lower right', facecolor=COLORS['bg_panel_raised'], labelcolor=COLORS['text_primary'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['ink_line'])
        plt.tight_layout()
        st.pyplot(fig)
        panel_end()

        # Confusion matrices
        st.subheader('Confusion Matrices')
        cols = st.columns(2)
        for i, (name, res) in enumerate(results.items()):
            with cols[i % 2]:
                panel_start(name)
                cm = res['confusion_matrix']
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='mako',
                            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'],
                            ax=ax, cbar=False, linewidths=1, linecolor=COLORS['bg_panel'])
                ax.set_title(f'{name}')
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
                st.pyplot(fig)
                panel_end()
    else:
        st.info('Please run train_model.py first or click Run Pipeline on ML Pipeline page.')

# ============================================================
# MODEL METRICS PAGE
# ============================================================
elif st.session_state.page == 'Model Metrics':
    st.header('📏 Model Evaluation Metrics')

    panel_start("Glossary")
    st.write('**Accuracy:** Overall correct predictions / total predictions')
    st.write('**Precision:** Of all predicted real news, how many are actually real')
    st.write('**Recall:** Of all actual real news, how many are correctly detected')
    st.write('**F1-Score:** Balance between Precision and Recall')
    panel_end()

    if os.path.exists('models/results.pkl'):
        results = load_results()
        for name, res in results.items():
            st.subheader(name)
            panel_start()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Accuracy', f"{res['accuracy']:.4f}")
            c2.metric('Precision', f"{res['precision']:.4f}")
            c3.metric('Recall', f"{res['recall']:.4f}")
            c4.metric('F1-Score', f"{res['f1']:.4f}")

            # Confusion matrix
            cm = res['confusion_matrix']
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='mako',
                        xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'],
                        ax=ax, cbar=False, linewidths=1, linecolor=COLORS['bg_panel'])
            ax.set_title(f'Confusion Matrix - {name}')
            st.pyplot(fig)
            panel_end()
    else:
        st.info('Please train the models first.')

# ============================================================
# ML PIPELINE PAGE
# ============================================================
elif st.session_state.page == 'ML Pipeline':
    st.header('🔧 ML Pipeline Implementation')
    steps = [
        ('1. Data Collection', 'Load the Fake News dataset from Kaggle, UCI, or any CSV file. The dataset must contain text and label columns.'),
        ('2. Data Preprocessing', 'Clean the text: convert to lowercase, remove numbers, remove punctuation, and remove extra spaces. Also handle missing values.'),
        ('3. Feature Engineering', 'Convert text into numbers using TF-IDF Vectorizer. It gives a score to each word based on its importance. We select top 5000 features.'),
        ('4. Train-Test Split', 'Split the dataset into 80% training and 20% testing using train_test_split() with random_state=42.'),
        ('5. Model Training', 'Train four ML models: Logistic Regression, Naive Bayes, Random Forest, and SVM.'),
        ('6. Model Evaluation', 'Evaluate each model using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.'),
        ('7. Model Comparison', 'Compare all models and identify the best performing one.')
    ]
    for title, desc in steps:
        with st.expander(title):
            st.write(desc)

    st.subheader('Pipeline Flow')
    panel_start()
    st.write('📁 Dataset → 🧹 Preprocessing → 🔢 TF-IDF Features → 🤖 ML Models → 📊 Evaluation → ⭐ Best Model')
    panel_end()

    st.subheader('Run Training Pipeline')
    st.markdown('<div class="pulse-cta">', unsafe_allow_html=True)
    run_clicked = st.button('▶️ Run / Retrain Models')
    st.markdown('</div>', unsafe_allow_html=True)
    if run_clicked:
        with st.spinner('Training models... please wait'):
            train_model.train_models()
        st.success('Training completed successfully!')

# ============================================================
# SOURCE CODE PAGE
# ============================================================
elif st.session_state.page == 'Source Code':
    st.header('💻 Source Code')
    files = ['app.py', 'train_model.py', 'requirements.txt']
    selected_file = st.selectbox('Select a file to view:', files)
    if os.path.exists(selected_file):
        with open(selected_file, 'r', encoding='utf-8') as f:
            code = f.read()
        st.code(code, language='python')
    else:
        st.error(f'{selected_file} not found.')
