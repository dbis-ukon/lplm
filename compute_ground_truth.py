import os
import sqlite3

def return_cardinality(query_list, c, dataset_size):
    """
    Calculate the conditional probability for a given LIKE pattern.

    Parameters:
    query_list (list of tuples): The LIKE pattern(s) to query in the database.
    c (sqlite3.Cursor): The cursor object for the SQLite database connection.
    dataset_size (int): The total size of the dataset.

    Returns:
    float: The conditional probability based on the LIKE pattern(s) provided.
    """
    if len(query_list) == 1:
        cn = c.execute('SELECT count(*) FROM column WHERE name LIKE ?', (query_list[0],)).fetchall()[0][0]
        prob = cn / dataset_size
        return prob
    else:
        cn = c.execute('SELECT count(*) FROM column WHERE name LIKE ?', (query_list[0],)).fetchall()[0][0]

        cn1 = c.execute('SELECT count(*) FROM column WHERE name LIKE ?', (query_list[1],)).fetchall()[0][0]
        if cn1 == 0:
            print(query_list)
        prob = float(cn) / cn1
        return prob


def find_all_possible_probabilities(like, all_con_prob_list):
    """
    Generate all possible conditional probabilities from the given LIKE pattern.

    Parameters:
    like (str): The transformed LIKE pattern.
    all_con_prob_list (list): The list to accumulate the conditional probabilities.

    Returns:
    list: A list of possible conditional probability queries.
    """

    wildcard_list = ['$', '^']
    if len(like) == 0:
        return all_con_prob_list
    elif len(like) == 1:
        all_con_prob_list.append(('%' + like[-1] + '%',))
        return all_con_prob_list
    else:
        if like[-1] not in wildcard_list:
            if like[-2] not in wildcard_list:
                all_con_prob_list.append(('%' + like + '%', '%' + like[:-1] + '%'))
                return find_all_possible_probabilities(like[:-1], all_con_prob_list)
            else:
                if len(like) > 2:
                    if like[-2] == '^':
                        all_con_prob_list.append(
                            ('%' + like[:-2] + '^' + like[-1] + '%', '%' + like[:-2] + '%' + like[-1] + '%'))
                        all_con_prob_list.append(('%' + like[:-2] + '%' + like[-1] + '%', '%' + like[:-2] + '%'))

                        return find_all_possible_probabilities(like[:-2], all_con_prob_list)
                    elif like[-2] == '$':
                        all_con_prob_list.append(('%' + like + '%', '%' + like[:-2] + '%' + like[-1] + '%'))
                        all_con_prob_list.append(('%' + like[:-2] + '%' + like[-1] + '%', '%' + like[:-2] + '%'))

                        return find_all_possible_probabilities(like[:-2], all_con_prob_list)
                else:
                    all_con_prob_list.append(('^' + like[-1] + '%', '%' + like[-1] + '%'))
                    all_con_prob_list.append(('%' + like[-1] + '%',))

                    return find_all_possible_probabilities('', all_con_prob_list)
        else:
            if len(all_con_prob_list) == 0:
                all_con_prob_list.append(('%' + like[:-1] + '^', '%' + like[:-1] + '%'))
                return find_all_possible_probabilities(like[:-1], all_con_prob_list)

def language_to_query(list_languages):
    """
    Convert the transformed LIKE patterns to SQL-compatible queries.

    Parameters:
    list_languages (list of lists): The transformed patterns.

    Returns:
    list of lists: SQL queries with appropriate wildcard replacements.
    """

    list_queries = []
    list_wildcards = ['$', '%', '_']
    for con in list_languages:
        list_pairs = []
        for l in con:
            query = ''
            l_ = l.replace('%^', '_').replace('^%', '_').replace('^', '_').replace('@', ' ')
            for i in range(len(l_)):
                if len(query) == 0:
                    query += l_[i]
                else:
                    if query[-1] not in list_wildcards and l_[i] not in list_wildcards:
                        query += '%' + l_[i]
                    else:
                        query += l_[i]
            list_pairs.append(query.replace('$', ''))
        list_queries.append(list_pairs)
    return list_queries


def LIKE_pattern_to_newLanguage(liste):
    """
    Transform the LIKE pattern into a new language format using special wildcards.

    Parameters:
    liste (list of str): The list of pattern segments that were split by '%' from the original LIKE pattern.

    Returns:
    str: A transformed pattern where each segment is adjusted and combined using the '^' and '$' wildcards.
    """
    transformed_pattern = ''
    for key in liste:
        if len(key) == 1:
            transformed_pattern += key
        else:
            new = ''
            count = 0
            for char in key:
                if count < 1:
                    new += char
                    count += 1
                else:
                    if new[-1] != '_' and char != '_' and char != '@' and new[-1] != '@':
                        new += '$' + char
                    else:
                        new += char
            transformed_pattern += new
    transformed_pattern = transformed_pattern.replace('_', '^')
    return transformed_pattern
def inject_type(liste, type_):
    """
    Injects the appropriate type of LIKE patterns.

    Parameters:
    liste (list): The list of transformed patterns.
    type_ (str): The type of modification to apply (e.g., 'prefix', 'suffix').

    Returns:
    list: The list with modified LIKE patterns based on the specified type.
    """
    newliste = []

    if type_ == 'prefix' or type_ == 'end_underscore':
        liste.insert(-1, [liste[-1][0][1:], liste[-1][0]])
        for i in range(len(liste)):
            if i < len(liste) - 2:
                newliste.insert(0, [liste[i][0][1:], liste[i][1][1:]])
            else:
                newliste.insert(0, liste[i])
        return newliste
    if type_ == 'suffix' or type_ == 'begin_underscore':
        liste.insert(0, [liste[0][0][:-1], liste[0][0]])
        for i in range(len(liste)):
            newliste.insert(0, liste[i])
        return newliste

    elif type_ == 'prefix_suffix':
        liste.insert(-1, [liste[-1][0][1:], liste[-1][0]])
        liste.insert(0, [liste[0][0][:-1], liste[0][0]])
        for i in range(len(liste)):
            if i < len(liste) - 2:
                newliste.insert(0, [liste[i][0][1:], liste[i][1][1:]])
            else:
                newliste.insert(0, liste[i])
        return newliste
    else:
        for i in range(len(liste)):
            newliste.insert(0, liste[i])
        return newliste



def main(db, listem, ground_truth_file_path, dataset_size):
    """
    Main function to compute the conditional probabilities for all LIKE patterns
    and save the results to a file.

    Parameters:
    db (str): Path to the SQLite database file.
    listem (list): The list of LIKE patterns.
    ground_truth_file_path (str): The path to save the output ground truth file.
    dataset_size (int): The total size of the dataset.
    """
    c = sqlite3.connect(db).cursor()

    file_to_save = open(ground_truth_file_path, 'w')
    for likepatterns in listem:
        print(likepatterns)
        newlike = likepatterns.strip().replace(' ', '@')
        likepatterns = ('%' + newlike + '%').replace('%%', '%').replace('%_', '_').replace('_%', '_')
        if likepatterns[0] == '%':
            likepatterns = likepatterns[1:]
        if likepatterns[-1] == '%':
            likepatterns = likepatterns[:-1]
        transformed_pattern = LIKE_pattern_to_newLanguage(likepatterns.split('%'))
        all_con_language_prob = find_all_possible_probabilities(transformed_pattern, [])
        all_con_prob1 = language_to_query(all_con_language_prob)
        if (newlike[0] == '%' and newlike[-1] != '%' and newlike[-1] != '_'):
            all_con_prob = inject_type(all_con_prob1, 'suffix')
            newlike = likepatterns + ':' + 'suffix'

        elif (newlike[0] != '%' and newlike[-1] == '%' and newlike[0] != '_'):
            all_con_prob = inject_type(all_con_prob1, 'prefix')
            newlike = likepatterns + ':' + 'prefix'

        elif newlike[0] != '%' and newlike[-1] != '%' and newlike[0] != '_' and newlike[-1] != '_':
            all_con_prob = inject_type(all_con_prob1, 'prefix_suffix')
            newlike = likepatterns + ':' + 'prefix_suffix'

        elif newlike[0] != '%' and newlike[-1] == '_' and newlike[0] != '_':
            all_con_prob = inject_type(all_con_prob1, 'end_underscore')
            newlike = likepatterns + ':' + 'end_underscore'

        elif newlike[-1] != '%' and newlike[0] == '_' and newlike[-1] != '_':
            all_con_prob = inject_type(all_con_prob1, 'begin_underscore')
            newlike = likepatterns + ':' + 'begin_underscore'
        else:
            all_con_prob = inject_type(all_con_prob1, 'substring')
            newlike = likepatterns + ':' + 'substring'
        liste = []
        try:
            for pair in all_con_prob:
                liste.append(return_cardinality(pair, c, dataset_size))

            s = [str(k) for k in liste]
            file_to_save.write(newlike + ':' + ' '.join(s) + '\n')
        except:
            print(likepatterns)


def load_like_patterns(filename):
    """
    Load LIKE patterns from a file into a list.

    Parameters:
    filename (str): The path to the file containing LIKE patterns, one per line.

    Returns:
    list: A list of LIKE patterns.
    """
    list_of_patterns = []
    with open(filename, 'r') as file:
        for line in file:
            list_of_patterns.append(line.strip())
    return list_of_patterns


if __name__ == "__main__":
    # Specify the paths required for running the script.

    # 1. Path to the SQLite database where the datasets are stored.
    db_path = 'sampleDB/author.db'  # Replace with the actual path to your SQLite database.

    # 2. Path to the file containing LIKE patterns that will be used for querying.
    like_patterns_path = 'author_names_training_set.txt'  # Replace with the path to the file containing the LIKE patterns.

    # 3. Path where the ground truth probabilities will be saved after computation.
    file_to_save_ground = 'author_names_training_set_groundtruth.txt'  # Replace with the desired path to save the output file.

    # Check if the file containing LIKE patterns exists.
    if not os.path.exists(like_patterns_path):
        # If the file does not exist, print an error message and stop.
        print(f"Error: Like patterns file not found at {like_patterns_path}")
    else:
        # If the file exists, load the LIKE patterns into a list.
        list_of_patterns = load_like_patterns(like_patterns_path)

        # Set the dataset size manually (this should be adjusted based on your data).
        datasetsize = 450000  # This is the size of the dataset (number of rows) in the database.

        # Call the main function to process the LIKE patterns and compute probabilities.
        # This will generate and save the ground truth output based on the patterns and dataset.
        main(db_path, list_of_patterns, file_to_save_ground, datasetsize)





