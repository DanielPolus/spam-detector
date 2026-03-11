from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import get_cleaned_data


# TF-IDF HERE


df = get_cleaned_data()

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['text_clean'])

print(X.shape)
print(vectorizer.get_feature_names_out()[:20])


# PREPAIRING DATA FOR TRAINING


le = LabelEncoder()
y = le.fit_transform(df['label'])
print(le.classes_) # must be ['ham', 'spam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train {X_train.shape[0]} examples")
print(f"Test {X_test.shape[0]} examples")


# TRAINING + CHECKING


models = {
    'Naive Bayes':         MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    results[name] = f1
    print(f"{name}: F1 = {f1:.4f}")

plt.figure(figsize=(8, 4))
plt.bar(results.keys(), results.values(), color=['steelblue', 'coral'])
plt.title('F1-score: Naive Bayes vs Logistic Regression')
plt.ylabel('F1 (spam)')
plt.ylim(0.85, 1.0)
plt.show()


# CONFUSION MATRICES


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['ham', 'spam'],
                yticklabels=['ham', 'spam'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {name}')

plt.tight_layout()
plt.show()