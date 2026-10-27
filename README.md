# Mental Health Statement Classification

An NLP project that classifies short text statements into one of seven mental health categories using traditional machine learning and deep learning models, and compares their performance.

**Classes:** Normal · Depression · Suicidal · Anxiety · Bipolar · Stress · Personality disorder

> ⚠️ **Disclaimer:** This is an educational/research project. It is **not** a diagnostic tool and must not be used to assess, diagnose, or make decisions about anyone's mental health. If you or someone you know is struggling, please contact a qualified professional or a local crisis line.

---

## Results

Test set: 13,171 statements (25% stratified hold-out).

| Model | Accuracy | Weighted F1 | Macro ROC-AUC |
|---|---|---|---|
| Majority Baseline | 0.3102 | 0.1469 | — |
| Naïve Bayes (tuned) | 0.6976 | 0.6970 | 0.9308 |
| Logistic Regression (tuned) | 0.7558 | 0.7553 | 0.9499 |
| BiLSTM | 0.7569 | 0.7567 | 0.9452 |
| **Transformer** | **0.7692** | **0.7661** | **0.9573** |

The custom Transformer performs best on every metric, with Logistic Regression a close and much cheaper second.

## Dataset

- **Source:** [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) (Kaggle)
- **File used:** `Combined Data.csv` (columns: `statement`, `status`)
- **Size:** 52,681 statements after dropping missing values

| Class | Count |
|---|---|
| Normal | 16,343 |
| Depression | 15,404 |
| Suicidal | 10,652 |
| Anxiety | 3,841 |
| Bipolar | 2,777 |
| Stress | 2,587 |
| Personality disorder | 1,077 |

The dataset is **not included** in this repository. Download it from Kaggle and place `Combined Data.csv` in the project root, next to the notebook.

## Project Structure

```
mental_health_classification/
├── mental_health_nlp_classification.ipynb   # full pipeline: EDA → preprocessing → models → evaluation
├── requirements.txt
├── README.md
└── Combined Data.csv                         # download separately (not tracked)
```

## Pipeline

1. **Exploratory Data Analysis** – class distribution, statement length analysis, word clouds per class.
2. **Text Preprocessing**
   - Lowercasing; removal of URLs, HTML tags, punctuation and digits
   - English stopword removal (NLTK)
   - Lemmatisation (spaCy `en_core_web_sm`)
3. **Train/Test Split** – 75/25 stratified split (`random_state=42`).
4. **Models**
   - **Majority-class baseline**
   - **Multinomial Naïve Bayes** – CountVectorizer (1–2 grams, 20k features) + TF-IDF, `alpha` tuned with 5-fold GridSearchCV
   - **Logistic Regression** – same features, class-balanced weights, `C` tuned with 5-fold GridSearchCV
   - **Bidirectional LSTM** – 128-dim embeddings, two stacked BiLSTM layers, dropout; compared against a 64-dim variant
   - **Transformer** – token + positional embeddings, two Transformer encoder blocks (8 heads), global average pooling; compared against a 4-head variant
5. **Evaluation** – accuracy, weighted F1, macro one-vs-rest ROC-AUC, confusion matrices and a per-class F1 heatmap.
6. **Inference** – a `predict_statement()` helper that applies the same preprocessing to raw text and returns the predicted class with a confidence score.

## Getting Started

**Requirements:** Python 3.11

```bash
# 1. Clone the repository
git clone https://github.com/GracieDataDiary/mental-health-classification.git
cd mental-health-classification

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Add the dataset
#    Download "Combined Data.csv" from Kaggle and place it in this folder

# 5. Launch the notebook
jupyter notebook mental_health_nlp_classification.ipynb
```

The NLTK stopword list downloads automatically when the notebook runs.

> **Note:** Lemmatising ~53k statements with spaCy takes a few minutes, and training the BiLSTM and Transformer models is noticeably faster on a GPU.

## Example Usage

After running the notebook end-to-end:

```python
predict_statement("I'm overwhelmed with work deadlines and can't seem to calm down.",
                  model_choice='transformer')
# {'predicted': 'Stress', 'confidence': '41.26%'}
```

`model_choice` can be `'nb'`, `'lr'`, `'bilstm'` or `'transformer'`.

## Limitations

- Overall accuracy is ~77%, and several hand-written test statements in the notebook are misclassified (for example, clearly distressed statements predicted as *Normal*). The models have learned dataset-specific patterns and do not generalise reliably to arbitrary text.
- Classes overlap heavily (e.g. Depression vs. Suicidal, Anxiety vs. Stress), and the minority classes (Personality disorder, Stress, Bipolar) are under-represented.
- Labels come from the source dataset and reflect how the original posts were categorised, not clinical diagnoses.

## Possible Improvements

- Fine-tune a pre-trained language model (e.g. BERT / RoBERTa / MentalBERT)
- Address class imbalance with resampling or class-weighted loss for the deep learning models
- Calibrate prediction confidence and add error analysis on misclassified examples

## Tech Stack

Python · pandas · NumPy · scikit-learn · TensorFlow / Keras · NLTK · spaCy · Matplotlib · Seaborn · WordCloud · Jupyter
