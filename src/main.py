import os
import numpy
import json
import pickle


class TransformerEncoder:
    def __init__(self):
        self.encoding_lookup_table = None
        self.decoding_lookup_table = None

        self.reserved_encodings = {
            "unknown_char": {
                "encoded": 0,
                "decoded": "<UNK>"
            },
        }

        self.reserved_encodings_count = len(self.reserved_encodings)

    def build_vocabulary(self, training_text: str) -> (dict[str, int], dict[int, str]):
        # create encoding lookup table
        self.encoding_lookup_table = {}
        for i, char in enumerate(sorted(set(training_text))):
            uid = i + self.reserved_encodings_count
            self.encoding_lookup_table[char] = uid

        # create decoding lookup table
        self.decoding_lookup_table = {}
        for char, uid in self.encoding_lookup_table.items():
            self.decoding_lookup_table[uid] = char

        # add reserved unknown character
        for name, data in self.reserved_encodings.items():
            uid = data["encoded"]
            char = data["decoded"]
            self.decoding_lookup_table[uid] = char


    def encode(self, text: str):
        encoded_training_text_vector = []
        for char in text:
            try:
                encoded_character = self.encoding_lookup_table[char]
            except KeyError as e:
                print(f"KeyError: character not found in the encoding lookup table: {e}")
                encoded_character = self.reserved_encodings["unknown_char"]["encoded"]

            encoded_training_text_vector.append(encoded_character)

        return encoded_training_text_vector

    def decode(self, encoded_text: list[int]):
        decoded_text = ""
        for char_uid in encoded_text:
            try:
                decoded_character = self.decoding_lookup_table[char_uid]
            except KeyError as e:
                print(f"KeyError: character not found in the decoding lookup table: {e}")
                unknown_char_id = self.reserved_encodings["unknown_char"]["encoded"]
                decoded_character = self.decoding_lookup_table[unknown_char_id]

            decoded_text += decoded_character

        return decoded_text

    def save_encodings(self, filepath: str):
        encodings = {
            "encoding_lookup_table": self.encoding_lookup_table,
            "decoding_lookup_table": self.decoding_lookup_table,
            "reserved_encodings": self.reserved_encodings
        }
        with open(filepath, "wb") as file:
            pickle.dump(encodings, file)

    def load_encodings(self, filepath: str):
        with open(filepath, "rb") as file:
            encodings = pickle.load(file)
            self.encoding_lookup_table = encodings["encoding_lookup_table"]
            self.decoding_lookup_table = encodings["decoding_lookup_table"]
            self.reserved_encodings = encodings["reserved_encodings"]


def main():
    working_dir = os.getcwd()
    training_dataset_path = os.path.join(working_dir, "data", "polish_novel.txt")
    user_data_dir = os.path.join(working_dir, "user_data")
    encodings_file = os.path.join(user_data_dir, "encodings_data.dat")

    transformer_encoder = TransformerEncoder()

    # load training text
    with open(training_dataset_path, "r", encoding="utf-8") as file:
        training_text = file.read()
        print(training_text)

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

    encoded_text = transformer_encoder.encode(training_text)
    print(encoded_text)

    decoded_text = transformer_encoder.decode(encoded_text)
    print(decoded_text)

    assert training_text == decoded_text



if __name__ == "__main__":
    main()