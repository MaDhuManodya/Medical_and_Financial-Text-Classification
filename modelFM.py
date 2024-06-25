import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import re

# Load dataset
df = pd.read_json(r"C:\Users\manod\Desktop\DCModel\News_Category_Dataset_v3.json\News_Category_Dataset_v3.json", lines=True)


# Relabel categories
def relabel_category(category):
    financial_keywords = ["BUSINESS", "MONEY"]
    medical_keywords = ["HEALTHY LIVING", "WELLNESS", "HEALTH", "MEDICAL"]
    if category in financial_keywords:
        return "Financial"
    elif category in medical_keywords:
        return "Medical"
    else:
        return "Other"

df['category'] = df['category'].apply(relabel_category)

# Filter out "Other" category for binary classification
df = df[df['category'] != 'Other']

# Combine 'headline' and 'short_description' into a single text field
df['text'] = df['headline'] + " " + df['short_description']

# Preprocess the text
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

df['text'] = df['text'].apply(clean_txt)

# Encode the labels
encoder = LabelEncoder()
df['category'] = encoder.fit_transform(df['category'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, stratify=df['category'])

# Train the model
text_clf = Pipeline([
    ('vect', CountVectorizer(stop_words="english")),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB(alpha=0.01)),
])

text_clf.fit(X_train, y_train)

# Save the model
with open('model_binary.pkl', 'wb') as f:
    pickle.dump(text_clf, f)

# Save the encoder
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
