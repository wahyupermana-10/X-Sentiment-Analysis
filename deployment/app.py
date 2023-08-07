import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from tensorflow .keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import warnings
warnings.filterwarnings(action='ignore')

st.header('Milestone 2 Phase 2')
st.write("Nama: Risqi Wahyu Permana")
st.write("Batch: HCK 006")

#load data

st.title('Prediction')
st.write('Please input your data')
#input data
text = st.text_input('Text')
st.write('Input text: ', text)

data = {
    'text': [text],
}
input = pd.DataFrame(data)

#load model
model = load_model('sentiment.h5')

#prediction
def predict(df):
    tokenizer = Tokenizer()
    x_data_val = pad_sequences(tokenizer.texts_to_sequences(df.text), maxlen = 30)
    x_scores = model.predict(x_data_val, verbose = 1, batch_size = 10000)
    df2 = pd.DataFrame(x_scores, columns=['Positive', 'Negative', 'Neutral'])
    new_df2 = pd.DataFrame(np.where(x_scores == np.max(x_scores, axis=1, keepdims=True), True, False), columns=['Positive_class', 'Negative_class', 'Neutral_class'])
    df2 = pd.concat([df2, new_df2], axis=1)
    condf2 = pd.concat([df, df2], axis=1)
    condf2['Sentiment'] = condf2[['Positive_class', 'Negative_class', 'Neutral_class']].idxmax(axis=1)
    condf2['Sentiment'] = condf2['Sentiment'].str.replace('_class', '')  # Hapus '_class' dari nama kategori
    condf2.Sentiment[0]
    return condf2.Sentiment[0]

if st.button('Predict'):
    st.write('Prediction: ', predict(input))