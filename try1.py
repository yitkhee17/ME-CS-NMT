# Import necessary libraries
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical

# Load data from file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read().split('\n')
    return data

source_sentences = load_data('source.txt')
target_sentences = load_data('target.txt')

# Tokenize the text (convert text to sequences of integers)
tokenizer_source = Tokenizer()
tokenizer_source.fit_on_texts(source_sentences)
source_sequences = tokenizer_source.texts_to_sequences(source_sentences)

tokenizer_target = Tokenizer()
tokenizer_target.fit_on_texts(target_sentences)
target_sequences = tokenizer_target.texts_to_sequences(target_sentences)

# Word to index mapping
source_index_word = tokenizer_source.index_word
target_index_word = tokenizer_target.index_word

# Determine max length for padding sequences
max_source_length = max([len(seq) for seq in source_sequences])
max_target_length = max([len(seq) for seq in target_sequences])

# Pad sequences to max length
source_data = pad_sequences(source_sequences, maxlen=max_source_length)
target_data = pad_sequences(target_sequences, maxlen=max_target_length)

# Load pre-trained word embeddings (GloVe)
embeddings_index = {}
with open('glove.6B.100d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Prepare input embedding matrix
embedding_dim = 100  # or whatever dimension GloVe you are using
source_vocab_size = len(tokenizer_source.word_index) + 1
embedding_matrix = np.zeros((source_vocab_size, embedding_dim))
for word, i in tokenizer_source.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# One-hot encode targets
target_data_one_hot = to_categorical(target_data)

# Define model
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(source_vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(target_vocab_size, embedding_dim)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding(decoder_inputs), initial_state=encoder_states)
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile
