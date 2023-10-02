import tensorflow as tf
import numpy as np
import os

"""## Read data"""

# Read text
path_to_file = 'C:/Users/juana/OneDrive - Universitas Airlangga/Dokumen/Kumpulan Tugas/Semester 5/Natural Language Processing/Autocomplete Algorithm/hp1.txt'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

"""## Process text: Build vocabulay and convert char to indices"""

# Build a vocabulary of unique characters in the text
vocab = sorted(set(text))

# Map each unique char to a different index
char2idx = {u: i for i, u in enumerate(vocab)}
# Map the index to the respective char
idx2char = np.array(vocab)
# Convert all the text to indices
text_as_int = np.array([char2idx[c] for c in text])

# Maximum length sentence we want for a single input
seq_length = 100

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

"""### Create input and target examples"""

def split_input_target(chunk):
    ''' Creates an input and target example for each sequence'''
    input_text = chunk[:-1]  # Removes the last character
    target_text = chunk[1:]  # Removes the first character
    return input_text, target_text

# Get inputs and targets for each sequence
dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

"""## Create training batches"""

# Batch size
BATCH_SIZE = 64
BUFFER_SIZE = 10000
# Suffle the dataset and get batches
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

"""## Build the model"""

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024


def generate_text(model, start_string, num_generate=1000, temperature=1.0):
    '''Generates text using the learned model'''

    # Converting our start string to numbers
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our result
    text_generated = []
    # Resets the state of metrics
    model.reset_states()

    for _ in range(num_generate):
        predictions = model(input_eval)
        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


