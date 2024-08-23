import streamlit as st
from keras import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
import numpy as np
import regex as re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Preprocessing function
stop_words = stopwords.words('english')
text_cleaning = "\b0\S*|\b[^A-Za-z0-9]+"

def preprocess_filter(text, stem=False):
    text = re.sub(text_cleaning, " ",str(text.lower()).strip())
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                stemmer = SnowballStemmer(language='english')
                token = stemmer.stem(token)
            tokens.append(token)
    return " ".join(tokens)

def word_embedding(text, vocab_size=5000):
    preprocessed_text = preprocess_filter(text)
    hot_encoded = one_hot(preprocessed_text, vocab_size)
    return hot_encoded

def create_model(embedded_features=40, max_length=42):
    model = Sequential()
    model.add(Embedding(5000, embedded_features, input_length=max_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load model
model = create_model()
model.load_weights('model_weights.h5')


max_length = 42

# Streamlit UI
st.title("Fake News Detection")
st.subheader("Enter the news headline to check if it's fake or not:")

input_text = st.text_input("News Headline")

if st.button("Predict"):
    if input_text:
        encoded = word_embedding(input_text)
        padded_encoded_title = pad_sequences([encoded], maxlen=max_length, padding='pre')
        prediction = model.predict(padded_encoded_title)
        output = np.where(prediction > 0.4, 1, 0)
        if output[0][0] == 1:
            st.write("**Prediction:** Yes, this news is fake.")
        else:
            st.write("**Prediction:** No, this news is not fake.")
    else:
        st.write("Please enter a news headline.")
