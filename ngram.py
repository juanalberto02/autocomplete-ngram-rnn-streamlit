import nltk
import random
import string
from bs4 import BeautifulSoup
import urllib.request
import re

def generate_suggestion(wikipedia_url, input_text, ngram_type='character', n=5, suggestion_length=100):
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^A-Za-z. ]', '', text)
        return text

    def generate_ngrams(text, n):
        ngrams = {}
        for i in range(len(text) - n):
            seq = text[i:i + n]
            ngrams.setdefault(seq, []).append(text[i + n])
        return ngrams

    def generate_word_ngrams(text, words):
        ngrams = {}
        words_tokens = nltk.word_tokenize(text)
        for i in range(len(words_tokens) - words):
            seq = ' '.join(words_tokens[i:i + words])
            ngrams.setdefault(seq, []).append(words_tokens[i + words])
        return ngrams

    def generate_suggestion_from_ngrams(ngrams, start_sequence, length):
        output = start_sequence
        for _ in range(length):
            if start_sequence not in ngrams:
                break
            possible_chars = ngrams[start_sequence]
            next_char = random.choice(possible_chars)
            output += next_char
            start_sequence = output[len(output) - n:len(output)]
        return output

    try:
        # Fetch the Wikipedia page
        raw_html = urllib.request.urlopen(wikipedia_url)
        raw_html = raw_html.read()

        # Parse the page and extract text
        article_html = BeautifulSoup(raw_html, 'html.parser')
        article_paragraphs = article_html.find_all('p')
        article_text = ''.join([para.text for para in article_paragraphs])

        # Clean the text
        article_text = clean_text(article_text)

        # Merge input_text with article_text
        input_text = clean_text(input_text)
        article_text = input_text + ' ' + article_text

        # Generate suggestions based on ngram_type
        if ngram_type == 'character':
            ngrams = generate_ngrams(article_text, n)
            start_sequence = input_text[:n]  # Reset start_sequence to match the new input_text
            suggestion = generate_suggestion_from_ngrams(ngrams, start_sequence, suggestion_length)
        elif ngram_type == 'word':
            ngrams = generate_word_ngrams(article_text, n)
            start_sequence = ' '.join(nltk.word_tokenize(input_text)[:n])  # Reset start_sequence to match the new input_text
            suggestion = generate_suggestion_from_ngrams(ngrams, start_sequence, suggestion_length)
        else:
            suggestion = "Invalid ngram_type. Use 'character' or 'word'."
    except Exception as e:
        suggestion = f"An error occurred: {str(e)}"

    return suggestion

# Example usage:
wikipedia_url = 'https://en.wikipedia.org/wiki/Deep_learning'
#input_text = input("Masukkan teks awalan: ")
#n = int(input("Masukkan jumlah n (jumlah kata dalam n-gram): "))
#suggestion = generate_suggestion(wikipedia_url, input_text, ngram_type='character', n=n, suggestion_length=100)
##print(suggestion)