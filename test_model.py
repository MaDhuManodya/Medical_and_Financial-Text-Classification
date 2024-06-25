import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import re
import string
import numpy as np

# Ensure the necessary NLTK data is downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define text preprocessing functions
stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()
my_sw = ['make', 'amp', 'news', 'new', 'time', 'u', 's', 'photos', 'get', 'say']

def black_txt(token):
    return token not in stop_words_ and token not in list(string.punctuation) and len(token) > 2 and token not in my_sw

def clean_txt(text):
    text = re.sub("'", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    clean_text = [wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
    return " ".join(clean_text)

# Load the model
with open('model_binary.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the encoder
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Example input text
input_text = 'Global stocks rose on Monday as investors anticipated a strong earnings season and looked forward to key economic data releases later this week. The S&P 500 and Nasdaq both reached record highs, driven by gains in technology and consumer discretionary stocks. Investors are particularly optimistic about earnings reports from major tech companies like Apple, Amazon, and Microsoft. Additionally, the Federal Reserve is set to meet later this week, and investors are keenly awaiting any hints about potential changes to monetary policy. Analysts expect the Fed to maintain its current stance, but any indications of future rate hikes could impact market sentiment. In the bond market, yields remained steady, reflecting investor confidence in the economic'

# Preprocess the input text
cleaned_text = clean_txt(input_text)

# Make a prediction
predicted = model.predict([cleaned_text])

# Ensure predicted is a one-dimensional array of integers
predicted = np.array(predicted).astype(int)

# Decode the prediction
predicted_category = encoder.inverse_transform(predicted)[0]

print(f'Predicted category: {predicted_category}')
