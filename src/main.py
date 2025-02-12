import os
import numpy

def main():
    # load the training data set
    working_dir = os.getcwd()
    training_data_set_path = os.path.join(working_dir, "data/polish_novel.txt")
    with open(training_data_set_path, "r", encoding="utf-8") as file:
        training_text = file.read()

    print(training_text)

    # create character to UID lookup table
    encoding_lookup_table = {}
    for uid, char in enumerate(set(training_text)):
        encoding_lookup_table[char] = uid

    # create UID to character lookup table
    decoding_lookup_table = {}
    for char, uid in encoding_lookup_table.items():
        decoding_lookup_table[uid] = char

    # encode the training text
    encoded_training_text_vector = [encoding_lookup_table[char] for char in training_text]
    print(encoded_training_text_vector)

    # decode the training text
    decoded_training_text = ""
    for char_uid in encoded_training_text_vector:
        decoded_character = decoding_lookup_table[char_uid]
        decoded_training_text += decoded_character

    print(decoded_training_text)


if __name__ == "__main__":
    main()