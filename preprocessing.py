import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from data_exploration import get_data

# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

def get_cleaned_data():
    df = get_data()
    df['text_clean'] = df['text'].apply(preprocess)
    return df

sample = "FREE WINNER!!! CALL NOW 88005553535!!!! CLAIM YOUR PRIZE"
print("Before:", sample)
print("After:", preprocess(sample))

if __name__ == "__main__":
    df = get_cleaned_data()