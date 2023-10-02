import streamlit as st
import tensorflow as tf
import numpy as np
import os
from ngram import generate_suggestion
from rnn import generate_text

# Fungsi untuk halaman N-gram
def ngram_page():
    st.title("Halaman N-gram")
    user_input = st.text_input("Masukkan teks :")
    
    char_count = st.number_input("Jumlah karakter yang ingin digenerate:", min_value=1)
    
    if st.button("Generate"):
        if not user_input:
            st.warning("Masukkan teks awalan.")
        else:
            suggestion = generate_suggestion('https://en.wikipedia.org/wiki/Deep_learning', user_input, ngram_type='character', n=char_count, suggestion_length=100)
            st.write("Teks yang dihasilkan:", suggestion)

# Fungsi untuk halaman RNN

# Define the path where you want to save the model
export_path = 'C:/Users/juana/OneDrive - Universitas Airlangga/Dokumen/Kumpulan Tugas/Semester 5/Natural Language Processing/Autocomplete Algorithm/model harry'

# Load the model
loaded_model = tf.keras.models.load_model(export_path)

def rnn_page():
    st.title("Halaman RNN")
    user_input = st.text_input("Masukkan teks :")
    char_count = st.number_input("Jumlah karakter yang ingin digenerate:", min_value=1)
    
    if st.button("Generate"):
        if not user_input:
            st.warning("Masukkan teks awalan.")
        else:
            # Tambahkan logika RNN di sini
            # Misalnya, Anda dapat menggunakan model RNN untuk menghasilkan teks yang baru
            predicted_text = generate_text(loaded_model, start_string=user_input, num_generate=char_count, temperature=1.0)
            st.write("Teks yang dihasilkan:", predicted_text)

# Main program
st.sidebar.title("Navbar")
page = st.sidebar.radio("Pilih Halaman:", ("N-gram", "RNN"))

if page == "N-gram":
    ngram_page()
elif page == "RNN":
    rnn_page()
