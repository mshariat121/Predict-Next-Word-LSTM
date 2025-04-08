import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "[Unknown]"

# Streamlit App Design
st.set_page_config(page_title="Next Word Predictor", page_icon="🔮", layout="centered")

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #6c63ff;'>🔮 Next Word Prediction App</h1>
        <p style='font-size: 18px;'>Predict the next word in a sequence using an LSTM model trained on Shakespeare's text.</p>
    </div>
""", unsafe_allow_html=True)

# Input box
input_text = st.text_input("Enter a sentence:", "To be or not to")

# Predict button
if st.button("Predict Next Word", type="primary"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    
    st.markdown(f"""
        <div style='text-align: center; padding: 20px;'>
            <h3 style='color: #4caf50;'>✨ Predicted Next Word: <span style='color: #2196f3;'>{next_word}</span></h3>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center;'>
        <p style='font-size: 14px; color: gray;'>Built with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)