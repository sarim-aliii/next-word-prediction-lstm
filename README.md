# Next Word Prediction using LSTM

A deep learning project that predicts the next word in a sequence using an LSTM-based language model. The model is trained on text data and deployed using a Streamlit web application for interactive predictions.

## Project Overview

This project demonstrates how Recurrent Neural Networks (RNNs), specifically **LSTM (Long Short-Term Memory)** networks, can be used to build a simple language model that predicts the next word in a sentence.

Example:

Input:
To be or not to

Prediction:
be

The application can also display multiple possible predictions with their confidence scores.

---

## Features

- LSTM-based text prediction model
- Interactive web interface using Streamlit
- Top-K next word predictions
- Fast model loading using Streamlit caching
- Clean and simple user interface
- Confidence score display for predicted words

---

## Project Structure

next-word-prediction-lstm/

app.py — Streamlit application  
next_word_lstm.h5 — Trained LSTM model  
tokenizer.pickle — Tokenizer used during training  
requirements.txt — Project dependencies  
README.md — Project documentation  

---

## Installation

1. Clone the repository

git clone https://github.com/sarim-aliii/next-word-prediction-lstm.git  
cd next-word-prediction-lstm

2. (Optional) Create a virtual environment

python -m venv venv

Activate environment:

Linux / Mac  
source venv/bin/activate

Windows  
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

---

## Run the Application

Start the Streamlit app:

streamlit run app.py

After running the command, open your browser and go to:

http://localhost:8501

---

## Model Architecture

The neural network consists of:

- Embedding Layer (converts words to dense vectors)
- LSTM Layer (captures sequential dependencies in text)
- Dense Output Layer with Softmax activation (predicts probability of next word)

Loss Function:
Categorical Crossentropy

Optimizer:
Adam

Early stopping is used during training to prevent overfitting.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Streamlit
- Pickle

---

## Example

Input:
Machine learning is

Output predictions:
fun  
powerful  
important

---

## Future Improvements

- Implement Transformer-based models (GPT style)
- Train on larger datasets
- Generate complete sentences instead of just the next word
- Deploy the application on Streamlit Cloud or HuggingFace Spaces
- Add real-time autocomplete functionality

---

## License

This project is open-source and available under the MIT License.

---

## Author

Ali  
Machine Learning & AI Enthusiast
