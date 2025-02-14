import math
import os
import argparse  # Add argparse import
from typing import Union
import numpy as np
import pickle


def initialize_random_weight_matrix(size):
    dim_sum = np.sum(size)
    limit = np.sqrt(6 / dim_sum)
    weight_matrix = np.random.uniform(-limit, limit, size)
    return weight_matrix


def softmax(input_matrix):
    return np.exp(input_matrix) / np.sum(np.exp(input_matrix), axis=1, keepdims=True)
    


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
            uid = int(i + self.reserved_encodings_count)
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

        self.vocabulary_size = len(self.char_encoding_lookup_table)

        # initialize embeddings_matrix with random values
        embedding_random_value_max = 1 / math.sqrt(self.embedding_dimension)
        embedding_random_value_min = -embedding_random_value_max

        self.embedding_matrix = np.random.uniform(
            low=embedding_random_value_min,
            high=embedding_random_value_max,
            size=(self.vocabulary_size, self.embedding_dimension)
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
    
    def encode_to_token_id(self, text: str):
        token_ids = np.zeros(len(text), dtype=int)
        for i, char in enumerate(text):
            try:
                char_id = self.char_encoding_lookup_table[char]
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
        weight_matrix_size = (embedding_dimension, embedding_dimension)

        self.W_q = initialize_random_weight_matrix(weight_matrix_size)
        self.W_k = initialize_random_weight_matrix(weight_matrix_size)
        self.W_v = initialize_random_weight_matrix(weight_matrix_size)

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
            raw_attention_scores /= np.sqrt(head_dim)  # apply scaling

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

        assert concatenated_attention_output.shape[-1] == embedding_dimension

        self.output = np.random.uniform(-0.1, 0.1, (embedding_dimension, embedding_dimension))
        self.output = np.dot(concatenated_attention_output, self.output)


class FeedForwardLayer():
    def __init__(self, self_attention_layer_output, embedding_dimension, input_sequence_length):
        self.hidden_dimension = embedding_dimension * 4

        self_attention_layer_output.reshape(1, input_sequence_length, embedding_dimension)

        W_1 = initialize_random_weight_matrix((embedding_dimension, self.hidden_dimension))
        b_1 = np.zeros(self.hidden_dimension)

        W_2 = initialize_random_weight_matrix((self.hidden_dimension, embedding_dimension))
        b_2 = np.zeros(embedding_dimension)

        self.H = np.dot(self_attention_layer_output, W_1) + b_1
        shape_H = self.H.shape
        
        # standard Gaussian cumulative distribution function (CDF) approximation
        CDF_distribution = np.full(shape_H, np.e)**-self.H
        H_GELU = self.H * 1.702 * (np.ones(shape_H) / (np.ones(shape_H) + CDF_distribution))

        self.output = np.dot(H_GELU, W_2) + b_2


class LayerNorm():
    def __init__(self, self_attention_layer_output, FFN_output):
        self.residual_connection = self_attention_layer_output + FFN_output # this should be computed before layer norm initialization and passed as input to it
        self.mean = np.mean(self.residual_connection, axis=-1, keepdims=True)
        self.variance = np.var(self.residual_connection, axis=-1, keepdims=True)
        epsilon = np.finfo(float).eps
        self.gamma = 1 # learnable scale parameter
        self.beta = 0 # learnable shift parameter

        self.output = (self.residual_connection - self.mean) / np.sqrt(self.variance + epsilon) * self.gamma + self.beta

def test_encoding_and_decoding(transformer_encoder: TransformerEncoder, training_text: str):
    encoded_text = transformer_encoder.encode(training_text, use_positional_encoding=False)
    decoded_text = transformer_encoder.decode(encoded_text)
    assert training_text == decoded_text


def main():
    parser = argparse.ArgumentParser(description="Transformer Encoder")
    parser.add_argument('-load_encodings', type=str, help='Path to the encodings file')
    args = parser.parse_args()

    working_dir = os.getcwd()
    training_dataset_path = os.path.join(working_dir, "data", "polish_novel.txt")
    user_data_dir = os.path.join(working_dir, "user_data")
    
    encodings_file = args.load_encodings if args.load_encodings else os.path.join(user_data_dir, "encodings_data.dat")

    transformer_encoder = TransformerEncoder()

    # load training text
    with open(training_dataset_path, "r", encoding="utf-8") as file:
        training_text = file.read()
        # print(training_text)

    # build or load encodings
    if args.load_encodings:
        print(f"Loading encodings from file {encodings_file}...")
        transformer_encoder.load_encodings(encodings_file)
    else:
        print("Building encodings from scratch...")
        transformer_encoder.build_vocabulary(training_text)
        transformer_encoder.save_encodings(encodings_file)

    # test encoding and decoding
    test_encoding_and_decoding(transformer_encoder, training_text)

    training_chunk_size = 128
    training_text_length = len(training_text)

    # generate training data chunks
    training_tokens_chunk_list = []
    for training_chunk_num in range(training_text_length // training_chunk_size):
        chunk_start = training_chunk_num*training_chunk_size
        chunk_end = min(chunk_start + training_chunk_size, training_text_length)
        
        text_chunk = training_text[chunk_start:chunk_end]
        training_tokens_chunk_list.append(text_chunk)

    for training_chunk_num, training_target_tokens in enumerate(training_tokens_chunk_list):
        # encode input to embeddings
        input_embeddings = transformer_encoder.encode(training_target_tokens)
        input_sequence_length = input_embeddings.shape[0]

        # build self-attention layer
        heads_count = 8
        embedding_dimension = transformer_encoder.get_embedding_dimension()

        self_attention_layer = SelfAttentionLayer(input_embeddings, embedding_dimension, heads_count)
        self_attention_layer_output = self_attention_layer.output
        
        # apply the position-wise feed-forward layer
        feedforward_layer = FeedForwardLayer(self_attention_layer_output, embedding_dimension, input_sequence_length)
        FFN_output = feedforward_layer.output
        
        # normalize layers
        layer_norm = LayerNorm(self_attention_layer_output, FFN_output)
        layer_norm_output = layer_norm.output

        # apply final linear transformation (projection to vocabulary space)
        vocab_size = transformer_encoder.vocabulary_size
        hidden_dimension = feedforward_layer.hidden_dimension

        W_vocab = initialize_random_weight_matrix((embedding_dimension, vocab_size))
        b_vocab = np.zeros(vocab_size)

        layer_norm_output.reshape(1, input_sequence_length, embedding_dimension)

        logits = np.dot(layer_norm_output, W_vocab) + b_vocab
        predicted_probabilities = softmax(logits)
        predicted_tokens = np.argmax(predicted_probabilities, axis=-1)

        training_target_token_id = transformer_encoder.encode_to_token_id(training_target_tokens[1:])
        
        if training_chunk_num + 1 > len(training_tokens_chunk_list):
            last_token = training_tokens_chunk_list[training_chunk_num + 1][0]
            last_token_id = transformer_encoder.encode_to_token_id(last_token)
            training_target_token_id = np.append(training_target_token_id, last_token_id)
        else:
            # discard last generated token without ground truth in data set
            predicted_tokens = predicted_tokens[:-1]

        predictions_num = predicted_tokens.shape[0]

        # losses = np.zeros(predictions_num)
        # for t in range(predictions_num):
        #     correct_id = training_target_token_id[t]

        #     p_correct = predicted_probabilities[t, correct_id]
        #     loss = -np.log(p_correct)
        #     losses[t] = loss

        # loss = np.mean(losses)

        epsilon = np.finfo(float).eps
        correct_probs = predicted_probabilities[np.arange(predictions_num), training_target_token_id]
        loss = -np.mean(np.log(correct_probs + epsilon))

        # prepare AdamW tensor buffers for backward pass
        W_q_mean = np.zeros((embedding_dimension, embedding_dimension))
        W_q_variance = np.zeros((embedding_dimension, embedding_dimension))
        W_k_mean = np.zeros((embedding_dimension, embedding_dimension))
        W_k_variance = np.zeros((embedding_dimension, embedding_dimension))
        W_v_mean = np.zeros((embedding_dimension, embedding_dimension))
        W_v_variance = np.zeros((embedding_dimension, embedding_dimension))
        W_vocab_mean = np.zeros((embedding_dimension, vocab_size))
        W_vocab_variance = np.zeros((embedding_dimension, vocab_size))


        # final linear transformation
        one_hot_ground_truth = np.zeros_like(predicted_probabilities)
        one_hot_ground_truth[np.arange(predictions_num), training_target_token_id] = 1

        gradient_loss_logits = predicted_probabilities - one_hot_ground_truth

        W_vocab = np.dot(layer_norm_output.T, gradient_loss_logits)
        b_vocab = np.sum(gradient_loss_logits, axis=0)

        # layer norm
        N = layer_norm.residual_connection.shape[1]

        gradient_loss_layer_norm_output = np.dot(gradient_loss_logits, W_vocab.T)
        gradient_loss_gamma = np.sum(gradient_loss_layer_norm_output * (layer_norm.residual_connection - layer_norm.mean) / (layer_norm.variance + epsilon), axis=0)
        gradient_loss_beta = np.sum(gradient_loss_layer_norm_output, axis=0)
        gradient_loss_h = (1.0 / N)


        pass
    pass

if __name__ == "__main__":
    main()