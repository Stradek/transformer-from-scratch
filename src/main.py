import math
import os
import argparse  # Add argparse import
from typing import Union
import numpy as np
import pickle

TRAINING_CHUNK_SIZE = 128
TRANSFORMER_EMBEDDING_DIMENSION_SIZE = 128
ENCODER_MAX_SEQUENCE_LENGTH = 2048
TRANSFORMER_HEADS_COUNT = 8
TRANSFORMER_HIDDEN_DIMENSION_SIZE = TRANSFORMER_EMBEDDING_DIMENSION_SIZE * 4

class TransformerEncoder:
    def __init__(self, max_sequence_length, embedding_dimension_size, training_text: str = ""):
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension_size = embedding_dimension_size

        self.reserved_encodings = {
            "unknown_char": {
                "encoded": 0,
                "decoded": "<UNK>"
            },
        }
        self.reserved_encodings_count = len(self.reserved_encodings)

        if training_text != "":
            self.char_encoding_table = self.build_character_encoding_lookup_table(training_text)
            self.char_decoding_table = self.build_character_decoding_lookup_table()
            self.add_reserved_characters_to_encoding_decoding_table()

            self.vocabulary_size = len(self.char_encoding_table)

            embedding_random_value_max = 1 / math.sqrt(self.embedding_dimension_size)
            embedding_random_value_min = -embedding_random_value_max

            self.embedding_matrix = np.random.uniform(
                low=embedding_random_value_min,
                high=embedding_random_value_max,
                size=(self.vocabulary_size, self.embedding_dimension_size)
            )

            # build embedding decoding lookup table
            self.embedding_decoding_lookup_table = {}
            for i, vector in enumerate(self.embedding_matrix):
                self.embedding_decoding_lookup_table[tuple(vector)] = i

            # build positional encoding matrix
            self.positional_encoding_matrix = np.zeros((self.max_sequence_length, self.embedding_dimension_size))
            for x in range(self.max_sequence_length):
                for y in range(self.embedding_dimension_size):
                    if y%2 == 0:
                        self.positional_encoding_matrix[x][y] = math.sin(x / (10000 ** (y / self.embedding_dimension_size)))
                    else:
                        self.positional_encoding_matrix[x][y] = math.cos(x / (10000 ** (y / self.embedding_dimension_size)))

    def add_reserved_characters_to_encoding_decoding_table(self):
        for name, data in self.reserved_encodings.items():
            uid = data["encoded"]
            char = data["decoded"]
            self.char_encoding_table[uid] = char
            self.char_decoding_table[uid] = char

    def build_character_encoding_lookup_table(self, training_text):
        char_encoding_table = {}
        for i, char in enumerate(sorted(set(training_text))):
            uid = int(i + self.reserved_encodings_count)
            char_encoding_table[char] = uid

        return char_encoding_table

    def build_character_decoding_lookup_table(self):
        char_decoding_table = {}
        for char, uid in self.char_encoding_table.items():
            char_decoding_table[uid] = char

        return char_decoding_table

    def get_vocabulary_size(self):
        return self.vocabulary_size

    def find_embedding_vector_id(self, embedding_vector: np.ndarray) -> int:
        return self.embedding_decoding_lookup_table.get(tuple(embedding_vector), -1)

    def encode(self, text: str, use_positional_encoding=True) -> np.ndarray:
        if use_positional_encoding:
            print("WARNING! Encoding with positional encoding.")
        else:
            print("WARNING! Encoding without positional encoding.")

        encoded_training_text = np.zeros((len(text), self.embedding_dimension_size))
        for i, char in enumerate(text):
            try:
                char_id = self.char_encoding_table[char]

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
    
    def encode_to_token_id(self, text: str):
        token_ids = np.zeros(len(text), dtype=int)
        for i, char in enumerate(text):
            try:
                char_id = self.char_encoding_table[char]
            except KeyError as e:
                print(f"KeyError: character not found in the encoding lookup table: {e}")
                char_id = self.reserved_encodings["unknown_char"]["encoded"]
            
            token_ids[i] = char_id

        return token_ids

    def decode(self, encoded_text: list[int]):
        decoded_text = ""
        for embedding_vector in encoded_text:
            try:
                char_id = self.find_embedding_vector_id(embedding_vector)
                decoded_character = self.char_decoding_table[char_id]
            except KeyError as e:
                print(f"KeyError: character not found in the decoding lookup table: {e}")
                unknown_char_id = self.reserved_encodings["unknown_char"]["encoded"]
                decoded_character = self.char_decoding_table[unknown_char_id]

            decoded_text += decoded_character

        return decoded_text

    def get_transformer_encoding_data(self):
        data = {
            "max_sequence_length": self.max_sequence_length,
            "encoding_lookup_table": self.char_encoding_table,
            "decoding_lookup_table": self.char_decoding_table,
            "vocabulary_size": self.vocabulary_size,
            "reserved_encodings": self.reserved_encodings,
            "reserved_encodings_count": self.reserved_encodings_count,
            "embedding_dimension_size": self.embedding_dimension_size,
            "embedding_matrix": self.embedding_matrix,
            "embedding_decoding_lookup_table": self.embedding_decoding_lookup_table,
            "positional_encoding_matrix": self.positional_encoding_matrix
        }
        return data

    def load_encodings(self, filepath: str):
        with open(filepath, "rb") as file:
            encodings = pickle.load(file)

            self.max_sequence_length = encodings["max_sequence_length"]
            self.char_encoding_table = encodings["encoding_lookup_table"]
            self.char_decoding_table = encodings["decoding_lookup_table"]
            self.vocabulary_size = encodings["vocabulary_size"]
            self.reserved_encodings = encodings["reserved_encodings"]
            self.reserved_encodings_count = encodings["reserved_encodings_count"]
            self.embedding_dimension_size = encodings["embedding_dimension_size"]
            self.embedding_matrix = encodings["embedding_matrix"]
            self.embedding_decoding_lookup_table = encodings["embedding_decoding_lookup_table"]
            self.positional_encoding_matrix = encodings["positional_encoding_matrix"]

class LayerNorm:
    def __init__(self):
        self.gamma = 1.0 # learnable scale parameter
        self.beta = 0.0 # learnable shift parameter
        self.epsilon = np.finfo(float).eps

    def forward(self, input_matrix):
        self.input = input_matrix

        self.mean = np.mean(input_matrix, axis=-1, keepdims=True)
        self.variance = np.var(input_matrix, axis=-1, keepdims=True)
        self.std = np.sqrt(self.variance + self.epsilon)
        self.norm = (input_matrix - self.mean) / self.std

        return self.gamma * self.norm + self.beta

    def backward(self, d_out):
        N = self.input.shape[-1]
        
        d_gamma = np.sum(d_out * self.norm, axis=0) 
        d_beta = np.sum(d_out, axis=0)
        d_norm = d_out * self.gamma
        d_input = (1.0 / N) / self.std * (N * d_norm - 
                                          np.sum(d_norm, axis=-1, keepdims=True) - 
                                          self.norm * np.sum(d_norm * self.norm, axis=-1, keepdims=True))
        
        return d_input, d_gamma, d_beta


class SelfAttentionLayer:
    def __init__(self, heads_count, embedding_dimension_size):
        self.heads_count = heads_count
        self.embedding_dimension_size = embedding_dimension_size
        self.head_dim = self.embedding_dimension_size // heads_count
        weight_matrix_size = (self.embedding_dimension_size, self.embedding_dimension_size)

        self.W_q = initialize_random_weight_matrix(weight_matrix_size)
        self.W_k = initialize_random_weight_matrix(weight_matrix_size)
        self.W_v = initialize_random_weight_matrix(weight_matrix_size)
        self.W_o = initialize_random_weight_matrix(weight_matrix_size)

    def forward(self, input_matrix):
        # compute attention scores
        Q = np.dot(input_matrix, self.W_q)
        K = np.dot(input_matrix, self.W_k)
        V = np.dot(input_matrix, self.W_v)

        Q = np.reshape(Q, (-1, self.heads_count, self.head_dim))
        K = np.reshape(K, (-1, self.heads_count, self.head_dim))
        V = np.reshape(V, (-1, self.heads_count, self.head_dim))

        Q_heads_list = np.split(Q, self.heads_count, axis=1)
        K_heads_list = np.split(K, self.heads_count, axis=1)
        V_heads_list = np.split(V, self.heads_count, axis=1)

        attention_outputs = []
        for i in range(self.heads_count):
            Q_head = np.squeeze(Q_heads_list[i], axis=1)
            K_head = np.squeeze(K_heads_list[i], axis=1)
            V_head = np.squeeze(V_heads_list[i], axis=1)
            raw_attention_scores = np.dot(Q_head, K_head.T)
            raw_attention_scores /= np.sqrt(self.head_dim)  # apply scaling

            # numerical stability trick
            max_raw_attention_scores = np.max(raw_attention_scores, axis=1, keepdims=True)
            stable_attention_scores = raw_attention_scores - max_raw_attention_scores

            attention_weights = softmax(stable_attention_scores)

            row_sums = np.sum(attention_weights, axis=1)
            row_ones = np.ones(attention_weights.shape[0])
            assert np.allclose(row_sums, row_ones)

            attention_output = np.dot(attention_weights, V_head)
            attention_outputs.append(attention_output)

        concatenated_attention_output = np.concatenate(attention_outputs, axis=-1)

        assert concatenated_attention_output.shape[-1] == self.embedding_dimension_size

        return np.dot(concatenated_attention_output, self.W_o)

class FeedForwardLayer: 
    def __init__(self, embedding_dimension_size, hidden_dimension_size):
        self.embedding_dimension_size = embedding_dimension_size
        self.hidden_dimension_size = hidden_dimension_size

        self.W_1 = initialize_random_weight_matrix((self.embedding_dimension_size, self.hidden_dimension_size))
        self.W_2 = initialize_random_weight_matrix((self.hidden_dimension_size, self.embedding_dimension_size))
        self.b_1 = np.zeros(self.hidden_dimension_size)
        self.b_2 = np.zeros(self.embedding_dimension_size)

    def forward(self, input_matrix):
        H = input_matrix @ self.W_1 + self.b_1

        H = np.reshape(H, (1, H.shape[0], H.shape[1]))
        H = self.apply_activation_function(H)
        H = np.squeeze(H, axis=0)

        return H @ self.W_2 + self.b_2

    def apply_activation_function(self, input_tensor):
        M_SQRT2 = 1.41421356237309504880
        M_2_SQRTPI = 1.12837916709551257390

        k_beta = M_SQRT2 * M_2_SQRTPI * 0.5
        k_kappa = 0.044715
        input_tensor_cube = input_tensor ** 3
        inner = k_beta * (input_tensor + k_kappa * input_tensor_cube)
        return input_tensor * 0.5 * (1 + np.tanh(inner))

class OutputProjectionLayer:
    def __init__(self, embedding_dimension_size, vocab_size):
        self.embedding_dimension_size = embedding_dimension_size

        self.W_vocab = initialize_random_weight_matrix((self.embedding_dimension_size, vocab_size))
        self.b_vocab = np.zeros(vocab_size)

    def forward(self, input_matrix):
        logits = np.dot(input_matrix, self.W_vocab) + self.b_vocab
        return logits

    @staticmethod
    def logits_to_predicted_token_probabilities(logits):
        return softmax(logits)

    @staticmethod
    def predicted_token_probabilities_to_tokens(token_probabilities):
        return np.argmax(token_probabilities, axis=-1)

def test_encoding_and_decoding(transformer_encoder: TransformerEncoder, training_text: str):
    encoded_text = transformer_encoder.encode(training_text, use_positional_encoding=False)
    decoded_text = transformer_encoder.decode(encoded_text)
    assert training_text == decoded_text

def export_transformer_encodings(encodings_file, transformer_encoder):
    encoding_data = transformer_encoder.get_transformer_encoding_data()
    with open(encodings_file, "wb") as file:
        pickle.dump(encoding_data, file)

def initialize_random_weight_matrix(size):
    dim_sum = np.sum(size)
    limit = np.sqrt(6 / dim_sum)
    weight_matrix = np.random.uniform(-limit, limit, size)
    return weight_matrix

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def split_training_text(training_text, training_chunk_size=128):
    training_text_length = len(training_text)

    training_tokens_chunk_list = []
    for training_chunk_index in range(training_text_length // training_chunk_size):
        chunk_start = training_chunk_index * training_chunk_size
        chunk_end = min(chunk_start + training_chunk_size, training_text_length)
        text_chunk = training_text[chunk_start:chunk_end]

        training_tokens_chunk_list.append(text_chunk)

    return training_tokens_chunk_list

def load_training_text(training_dataset_path):
    with open(training_dataset_path, "r", encoding="utf-8") as file:
        training_text = file.read()
    
    return training_text

def main():
    parser = argparse.ArgumentParser(description="Transformer Encoder")
    parser.add_argument('-load_encodings', type=str, help='Path to the encodings file')
    args = parser.parse_args()

    working_dir = os.getcwd()
    training_dataset_path = os.path.join(working_dir, "data", "polish_novel.txt")
    user_data_dir = os.path.join(working_dir, "user_data")
    encodings_file = args.load_encodings if args.load_encodings else os.path.join(user_data_dir, "encodings_data.dat")

    training_text = load_training_text(training_dataset_path)

    # build or load encodings
    if args.load_encodings:
        print(f"Loading encodings from file {encodings_file}...")
        transformer_encoder = TransformerEncoder(ENCODER_MAX_SEQUENCE_LENGTH, TRANSFORMER_EMBEDDING_DIMENSION_SIZE)
        transformer_encoder.load_encodings(encodings_file)
    else:
        print("Building encodings from scratch...")
        transformer_encoder = TransformerEncoder(ENCODER_MAX_SEQUENCE_LENGTH, TRANSFORMER_EMBEDDING_DIMENSION_SIZE, 
                                                 training_text=training_text)

        export_transformer_encodings(encodings_file, transformer_encoder)

    test_encoding_and_decoding(transformer_encoder, training_text)

    vocab_size = transformer_encoder.get_vocabulary_size()

    layer_norm = LayerNorm()
    self_attention_layer = SelfAttentionLayer(TRANSFORMER_HEADS_COUNT, TRANSFORMER_EMBEDDING_DIMENSION_SIZE)
    feedforward_layer = FeedForwardLayer(TRANSFORMER_EMBEDDING_DIMENSION_SIZE, TRANSFORMER_HIDDEN_DIMENSION_SIZE)
    output_projection_layer = OutputProjectionLayer(TRANSFORMER_EMBEDDING_DIMENSION_SIZE, vocab_size)

    # Test LayerNorm forward and backward pass
    test_input = np.random.rand(4, TRANSFORMER_EMBEDDING_DIMENSION_SIZE)
    original_gamma = layer_norm.gamma
    original_beta = layer_norm.beta

    norm_output = layer_norm.forward(test_input)

    d_out = np.random.rand(*norm_output.shape)
    d_input, d_gamma, d_beta = layer_norm.backward(d_out)

    # Compare input, gamma, and beta
    assert np.allclose(test_input, layer_norm.input), "Input mismatch after forward pass."
    assert np.isclose(original_gamma, layer_norm.gamma), "Gamma mismatch after forward pass."
    assert np.isclose(original_beta, layer_norm.beta), "Beta mismatch after forward pass."

    print("LayerNorm forward and backward pass test completed successfully.")

    training_tokens_chunk_list = split_training_text(training_text, TRAINING_CHUNK_SIZE)

    for training_chunk_index, training_target_tokens in enumerate(training_tokens_chunk_list):
        input_embeddings = transformer_encoder.encode(training_target_tokens)

        self_attention_layer_output = self_attention_layer.forward(input_embeddings)
        # self_attention_layer_output = layer_norm.forward(self_attention_layer_output)
        
        ffn_output = feedforward_layer.forward(self_attention_layer_output)
        # ffn_output = layer_norm.forward(self_attention_layer_output)

        logits = output_projection_layer.forward(ffn_output)
        predicted_token_probabilities = OutputProjectionLayer.logits_to_predicted_token_probabilities(logits)
        predicted_tokens = OutputProjectionLayer.predicted_token_probabilities_to_tokens(logits)

        #############################
        # training
        #############################
        target_tokens = training_target_tokens[1:]
        training_target_token_id = transformer_encoder.encode_to_token_id(target_tokens)
        mask = np.ones(predicted_token_probabilities.shape[0])  
        predictions_num = predicted_tokens.shape[0]

        if training_chunk_index < len(training_tokens_chunk_list) - 1:
            # add first token from next batch to target_tokens
            # it's because when we input N tokens with id's 0, 1, 2, ..., n+127,
            # we get predictions for next token after input token.
            # that means we can find corresponding token id's 1, 2, 3, ..., 128, where 
            # token with id 128 is token we didn't pass as input.
            # ground truth for predicted token with id 128 is the token from next batch with id 0.

            next_batch_token = training_tokens_chunk_list[training_chunk_index + 1][0]
            next_batch_token_id = transformer_encoder.encode_to_token_id(next_batch_token)
            training_target_token_id = np.append(training_target_token_id, next_batch_token_id)
        else:
            # last batch 
            # hack: add null token and mask it out
            
            training_target_token_id = np.append(training_target_token_id, 0)
            mask[-1] = 0

        one_hot_ground_truth = np.zeros(predicted_token_probabilities.shape)
        one_hot_ground_truth[np.arange(predictions_num), training_target_token_id] = 1
        
        epsilon = np.finfo(float).eps

        loss = -np.mean(np.sum(one_hot_ground_truth * np.log(predicted_token_probabilities + epsilon)))

        # # prepare AdamW tensor buffers for backward pass
        # W_q_mean = np.zeros((embedding_dimension, embedding_dimension))
        # W_q_variance = np.zeros((embedding_dimension, embedding_dimension))
        # W_k_mean = np.zeros((embedding_dimension, embedding_dimension))
        # W_k_variance = np.zeros((embedding_dimension, embedding_dimension))
        # W_v_mean = np.zeros((embedding_dimension, embedding_dimension))
        # W_v_variance = np.zeros((embedding_dimension, embedding_dimension))
        # W_vocab_mean = np.zeros((embedding_dimension, vocab_size))
        # W_vocab_variance = np.zeros((embedding_dimension, vocab_size))
        
        pass
    pass

if __name__ == "__main__":
    main()
