




import random
import re

def generate_like_patterns(dataset_file_path, training_data_file, train_data_size):
    """
    Generate a set of random LIKE patterns from a given dataset and save them to a file.

    Parameters:
        dataset_file_path (str): The file path of the dataset containing rows to generate patterns from.
        training_data_file (str): The file path to save the generated patterns.
        train_data_size (int): The number of unique LIKE patterns to generate.
    """
    with open(dataset_file_path, 'r') as file:
        all_rows = [line.strip() for line in file]

    all_patterns = set()
    while len(all_patterns) < train_data_size:
        select_random_row = list(random.choice(all_rows))
        random_indexes = random.sample(range(len(select_random_row)), random.randint(0, len(select_random_row)))
        for key in random_indexes:
            if key + 1 not in random_indexes and key - 1 not in random_indexes:
                select_random_row[key] = '_'
            else:
                select_random_row[key] = '%'

        like_pattern = re.sub(r'%+', '%', ''.join(select_random_row)).replace('_%', '%').replace('%_', '%')
        if like_pattern and like_pattern != '%':
            all_patterns.add(like_pattern)

    with open(training_data_file, 'w') as saving_path:
        saving_path.write('\n'.join(all_patterns) + '\n')

if __name__ == "__main__":
    train_dataset_size = 50000
    generate_like_patterns('Datasets/author_name.csv',
                           'author_names_training_set.txt', train_dataset_size)