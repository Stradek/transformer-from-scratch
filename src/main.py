import os

def main():
    # Load the training data set
    working_dir = os.getcwd()
    training_data_set_path = os.path.join(working_dir, "data/polish_novel.txt")
    with open(training_data_set_path, "r", encoding="utf-8") as file:
        training_text = file.read()

    print(training_text)


if __name__ == "__main__":
    main()