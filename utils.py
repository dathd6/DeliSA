import json
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Streaming the json file for reading
def stream_json(json_file, chunk_size=None):
    with open(json_file, 'r') as f:
        chunk = []
        # Number of rows to read from the json file
        for line in f:
            chunk.append(json.loads(line))
            # Add chunk size for testing purposes
            if chunk_size and len(chunk) == chunk_size:
                return chunk

        if chunk:
            return chunk

def assign_sentiment(rating):
    # Function to assign sentiment based on star rating
    if rating >= 4:
        return 1  # Positive sentiment
    elif rating <= 3:
        return 0 # Negative sentiment
    return None  # Invalid rating value

def preprocess_text(text):
    # Tokenize and remove punctuation
    tokens = [word for word in word_tokenize(text) if word.isalnum()]
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)