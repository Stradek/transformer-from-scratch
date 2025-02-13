import math
import os
from typing import Union
import numpy as np
import pickle


def initialize_random_weight_matrix(min_value, max_value, size):
    weight_matrix = np.random.uniform(
            low = min_value,
            high = max_value,
            size = size
        )
    return weight_matrix


class TransformerEncoder:
    def __init__(self):
        self.embedding_dimension = 128
        self.max_sequence_length = 2048

        self.char_encoding_lookup_table = None
        self.char_decoding_lookup_table = None
        self.embedding_decoding_lookup_table = None
        self.embedding_matrix = None
        self.positional_encoding_matrix = None

        self.reserved_encodings = {
            "unknown_char": {
                "encoded": 0,
                "decoded": "<UNK>"
            },
        }
        self.reserved_encodings_count = len(self.reserved_encodings)

    def get_embedding_dimension(self):
        return self.embedding_dimension
    

    def get_max_sequence_length(self):
        return self.max_sequence_length
    

    def build_vocabulary(self, training_text: str) -> Union[dict[str, int], dict[int, str]]:
        # build encoding lookup table
        self.char_encoding_lookup_table = {}
        for i, char in enumerate(sorted(set(training_text))):
            uid = i + self.reserved_encodings_count
            self.char_encoding_lookup_table[char] = uid

        # build decoding lookup table
        self.char_decoding_lookup_table = {}
        for char, uid in self.char_encoding_lookup_table.items():
            self.char_decoding_lookup_table[uid] = char

        # add reserved unknown character
        for name, data in self.reserved_encodings.items():
            uid = data["encoded"]
            char = data["decoded"]
            self.char_encoding_lookup_table[uid] = char
            self.char_decoding_lookup_table[uid] = char

        vocabulary_size = len(self.char_encoding_lookup_table)

        # initialize embeddings_matrix with random values
        embedding_random_value_max = 1 / math.sqrt(self.embedding_dimension)
        embedding_random_value_min = -embedding_random_value_max

        self.embedding_matrix = np.random.uniform(
            low=embedding_random_value_min,
            high=embedding_random_value_max,
            size=(vocabulary_size, self.embedding_dimension)
        )

        # build embedding decoding lookup table
        self.embedding_decoding_lookup_table = {}
        for i, vector in enumerate(self.embedding_matrix):
            self.embedding_decoding_lookup_table[tuple(vector)] = i

        # build positional encoding matrix
        self.positional_encoding_matrix = np.zeros((self.max_sequence_length, self.embedding_dimension))
        for x in range(self.max_sequence_length):
            for y in range(self.embedding_dimension):
                if y%2 == 0:
                    self.positional_encoding_matrix[x][y] = math.sin(x / (10000 ** (y / self.embedding_dimension)))
                else:
                    self.positional_encoding_matrix[x][y] = math.cos(x / (10000 ** (y / self.embedding_dimension)))


    def get_vocabulary_size(self):
        return len(self.char_encoding_lookup_table)


    def find_embedding_vector_id(self, embedding_vector: np.ndarray) -> int:
        return self.embedding_decoding_lookup_table.get(tuple(embedding_vector), -1)


    def encode(self, text: str, use_positional_encoding=True) -> np.ndarray:
        if use_positional_encoding:
            print("WARNING! Encoding with positional encoding.")
        else:
            print("WARNING! Encoding without positional encoding.")

        encoded_training_text = np.zeros((len(text), self.embedding_dimension))
        for i, char in enumerate(text):
            try:
                char_id = self.char_encoding_lookup_table[char]

                embedding_vector = self.embedding_matrix[char_id] 
                if use_positional_encoding:
                    embedding_vector += self.positional_encoding_matrix[i]
            except KeyError as e:
                print(f"KeyError: character not found in the encoding lookup table: {e}")
                unknown_char_id = self.reserved_encodings["unknown_char"]["encoded"]
                
                embedding_vector = self.embedding_matrix[unknown_char_id]
                if use_positional_encoding:
                    embedding_vector += self.positional_encoding_matrix[i]

            encoded_training_text[i] = embedding_vector

        return encoded_training_text

    def decode(self, encoded_text: list[int]):
        decoded_text = ""
        for embedding_vector in encoded_text:
            try:
                char_id = self.find_embedding_vector_id(embedding_vector)
                decoded_character = self.char_decoding_lookup_table[char_id]
            except KeyError as e:
                print(f"KeyError: character not found in the decoding lookup table: {e}")
                unknown_char_id = self.reserved_encodings["unknown_char"]["encoded"]
                decoded_character = self.char_decoding_lookup_table[unknown_char_id]

            decoded_text += decoded_character

        return decoded_text

    def save_encodings(self, filepath: str):
        encodings = {
            "encoding_lookup_table": self.char_encoding_lookup_table,
            "decoding_lookup_table": self.char_decoding_lookup_table,
            "reserved_encodings": self.reserved_encodings
        }
        with open(filepath, "wb") as file:
            pickle.dump(encodings, file)

    def load_encodings(self, filepath: str):
        with open(filepath, "rb") as file:
            encodings = pickle.load(file)
            self.char_encoding_lookup_table = encodings["encoding_lookup_table"]
            self.char_decoding_lookup_table = encodings["decoding_lookup_table"]
            self.reserved_encodings = encodings["reserved_encodings"]

class SelfAttentionLayer():
    def __init__(self, input_embeddings, embedding_dimension, heads_count):
        head_dim = embedding_dimension // heads_count

        weights_random_max = math.sqrt(6) / embedding_dimension
        weight_matrix_size = (embedding_dimension, embedding_dimension)

        self.W_q = initialize_random_weight_matrix(-weights_random_max, weights_random_max, weight_matrix_size)
        self.W_k = initialize_random_weight_matrix(-weights_random_max, weights_random_max, weight_matrix_size)
        self.W_v = initialize_random_weight_matrix(-weights_random_max, weights_random_max, weight_matrix_size)

        # compute attention scores
        Q_heads = np.dot(input_embeddings, self.W_q)
        K_heads = np.dot(input_embeddings, self.W_k)
        V_heads = np.dot(input_embeddings, self.W_v)

        Q_heads = np.reshape(Q_heads, (-1, heads_count, head_dim))
        K_heads = np.reshape(K_heads, (-1, heads_count, head_dim))
        V_heads = np.reshape(V_heads, (-1, heads_count, head_dim))

        Q_heads_list = np.split(Q_heads, heads_count, axis=1)
        K_heads_list = np.split(K_heads, heads_count, axis=1)
        V_heads_list = np.split(V_heads, heads_count, axis=1)

        attention_outputs = []
        for i in range(heads_count):
            Q_head = np.squeeze(Q_heads_list[i], axis=1)
            K_head = np.squeeze(K_heads_list[i], axis=1)
            V_head = np.squeeze(V_heads_list[i], axis=1)
            raw_attention_scores = np.dot(Q_head, K_head.T)
            raw_attention_scores /= np.sqrt(head_dim) # apply scaling

            attention_weights = np.exp(raw_attention_scores) / np.sum(np.exp(raw_attention_scores), axis=1, keepdims=True)

            row_sums = np.sum(attention_weights, axis=1)
            assert np.allclose(row_sums, np.ones(attention_weights.shape[0]))

            attention_output = np.dot(attention_weights, V_head)
            attention_outputs.append(attention_output)

        concatenated_attention_output = np.concatenate(attention_outputs, axis=-1)

        assert concatenated_attention_output.shape[-1] == embedding_dimension

        self.W_o = np.random.uniform(-0.1, 0.1, (embedding_dimension, embedding_dimension))
        self.W_o = np.dot(concatenated_attention_output, self.W_o)


def test_encoding_and_decoding(transformer_encoder: TransformerEncoder, training_text: str):
    encoded_text = transformer_encoder.encode(training_text, use_positional_encoding=False)
    decoded_text = transformer_encoder.decode(encoded_text)
    assert training_text == decoded_text


def main():
    working_dir = os.getcwd()
    training_dataset_path = os.path.join(working_dir, "data", "polish_novel.txt")
    user_data_dir = os.path.join(working_dir, "user_data")
    encodings_file = os.path.join(user_data_dir, "encodings_data.dat")

    transformer_encoder = TransformerEncoder()

    # load training text
    with open(training_dataset_path, "r", encoding="utf-8") as file:
        training_text = file.read()
        # print(training_text)

    if os.path.exists(encodings_file):
        load_encodings_file = input("Do you want to load existing encodings file? (y/n) ") == "y"
    else:
        load_encodings_file = False

    # build or load encodings
    if load_encodings_file:
        print(f"Loading encodings from file {encodings_file}...")
        transformer_encoder.load_encodings(encodings_file)
    else:
        print("Building encodings from scratch...")
        transformer_encoder.build_vocabulary(training_text)
        transformer_encoder.save_encodings(encodings_file)

    # test encoding and decoding
    test_encoding_and_decoding(transformer_encoder, training_text)

    # encode input to embeddings
    input_embeddings = transformer_encoder.encode(training_text)
    input_sequence_length = input_embeddings.shape[0]

    # build self-attention layer
    heads_count = 8
    embedding_dimension = transformer_encoder.get_embedding_dimension()

    self_attention_layer = SelfAttentionLayer(input_embeddings, embedding_dimension, heads_count)
    batch_size = 128
    self_attention_layer.W_o.reshape(1, input_sequence_length, embedding_dimension)
    pass

if __name__ == "__main__":
    main()