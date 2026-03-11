# SMS Spam Detector

A classical NLP pipeline that classifies SMS messages as **spam** or **ham** (not spam).  
Built with Python, scikit-learn, and NLTK — no deep learning, no transformers — just solid fundamentals.

---

## Results

| Metric | Ham | Spam |
|---|---|---|
| Precision | 1.00 | 0.76 |
| Recall | 0.96 | 0.99 |
| F1-score | 0.98 | 0.86 |

**Overall accuracy: 97%**

> The dataset is imbalanced (~87% ham, ~13% spam), so F1-score is a more meaningful metric than accuracy here.  
> The model prioritizes high recall on spam (0.99) — nearly all spam is caught — at the cost of some false positives.

---

## Dataset

[SMS Spam Collection — UCI / Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

- 5,572 SMS messages
- 4,825 ham (87%) / 747 spam (13%)

---

## Pipeline

```
Raw text
   ↓ lowercase + remove punctuation & digits
   ↓ remove stopwords
   ↓ lemmatization (NLTK)
   ↓ TF-IDF vectorization (top 5000 features)
   ↓ Multinomial Naive Bayes
Spam / Ham
```

---

## Project Structure

```
spam-detector/
├── README.md
├── spam.csv
├── data_exploration.py    ← load & inspect the dataset
├── preprocessing.py       ← text cleaning pipeline
└── baseline_model.py      ← TF-IDF + Naive Bayes + evaluation
```

---

## How to Run

```bash
pip install scikit-learn nltk pandas matplotlib seaborn
python baseline_model.py
```

On first run, uncomment these two lines in `preprocessing.py`:

```python
# nltk.download('stopwords')
# nltk.download('wordnet')
```

---

## Tech Stack

- Python 3.9
- scikit-learn
- NLTK
- pandas
- matplotlib / seaborn
