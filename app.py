import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Model with Error Handling
@st.cache_resource
def load_lstm_model():
    if not os.path.exists("next_word_lstm.h5"):
        raise FileNotFoundError("next_word_lstm.h5 not found.")
    return load_model("next_word_lstm.h5")

# Load Tokenizer & Build Dictionary with Error Handling
@st.cache_resource
def load_tokenizer_and_mapping():
    if not os.path.exists("tokenizer.pickle"):
        raise FileNotFoundError("tokenizer.pickle not found.")
    
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
        
    # Reverse word index built inside cache for faster lookup
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    return tokenizer, index_to_word

# App Title & Description
st.title("Next Word Prediction using LSTM")
st.write("Enter a sequence of words and the model will predict the next word.")

# Attempt to load resources, stop execution if files are missing
try:
    model = load_lstm_model()
    tokenizer, index_to_word = load_tokenizer_and_mapping()
except FileNotFoundError as e:
    st.error(f"Error: {e} Please ensure the model has been trained and the required files are in the directory.")
    st.stop()

# Prediction Function
def predict_next_words(model, tokenizer, index_to_word, text, sequence_len, top_k=3):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return [("No known words found", 0.0)]

    # Simplified length logic
    if len(token_list) >= sequence_len:
        token_list = token_list[-sequence_len:]

    token_list = pad_sequences(
        [token_list],
        maxlen=sequence_len,
        padding="pre"
    )

    predictions = model.predict(token_list, verbose=0)[0]

    # Top K predictions
    top_indices = predictions.argsort()[-top_k:][::-1]

    words = []
    for i in top_indices:
        word = index_to_word.get(i, "")
        prob = predictions[i]
        words.append((word, prob))

    return words

# User Inputs
input_text = st.text_input(
    "Enter text",
    "To be or not to"
)

top_k = st.slider(
    "Number of predictions",
    1,
    5,
    3
)

# Execution
if st.button("Predict"):
    # Using exact input shape
    sequence_len = model.input_shape[1]

    with st.status("Running prediction...", expanded=True) as status:

        st.write("Processing input text...")
        predictions = predict_next_words(
            model,
            tokenizer,
            index_to_word,
            input_text,
            sequence_len,
            top_k
        )

        st.write("Generating predictions...")
        status.update(label="Prediction complete!", state="complete")

    st.subheader("Predicted Next Words")

    for word, prob in predictions:
        st.write(f"**{word}** (confidence: {prob:.4f})")