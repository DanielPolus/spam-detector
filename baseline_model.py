from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import get_cleaned_data


# TF-IDF HERE


df = get_cleaned_data()

vectorizer = TfidfVectorizer(max_features=5000)
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


nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

print(classification_report(y_pred, y_test, target_names=['ham', 'spam'])) # CLASSIFICATION REPORT

cm = confusion_matrix(y_test, y_pred) # CONFUSION MATRIX
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['ham', 'spam'],
            yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
