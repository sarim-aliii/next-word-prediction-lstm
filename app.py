import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


@st.cache_resource
def load_lstm_model():
    return load_model("next_word_lstm.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pickle", "rb") as handle:
        return pickle.load(handle)


model = load_lstm_model()
tokenizer = load_tokenizer()

# Reverse word index for faster lookup
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# Prediction
def predict_next_words(model, tokenizer, text, max_sequence_len, top_k=3):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return ["No known words found"]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
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


# App
st.title("Next Word Prediction using LSTM")
st.write("Enter a sequence of words and the model will predict the next word.")

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

if st.button("Predict"):
    max_sequence_len = model.input_shape[1] + 1

    with st.status("Running prediction...", expanded=True) as status:

        st.write("Processing input text...")
        predictions = predict_next_words(
            model,
            tokenizer,
            input_text,
            max_sequence_len,
            top_k
        )

        st.write("Generating predictions...")
        status.update(label="Prediction complete!", state="complete")

    st.subheader("Predicted Next Words")

    for word, prob in predictions:
        st.write(f"**{word}**  (confidence: {prob:.4f})")