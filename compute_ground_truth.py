import os
import sqlite3

def return_cardinality(query_list, c, dataset_size):
    if len(query_list) == 1:
        cn = c.execute('SELECT count(*) FROM pattern WHERE trans LIKE ?', (query_list[0],)).fetchall()[0][0]
        prob = cn / dataset_size
        return prob
    else:
        cn = c.execute('SELECT count(*) FROM pattern WHERE trans LIKE ?', (query_list[0],)).fetchall()[0][0]

        cn1 = c.execute('SELECT count(*) FROM pattern WHERE trans LIKE ?', (query_list[1],)).fetchall()[0][0]
        if cn1 == 0:
            print(query_list)
        prob = float(cn) / cn1
        return prob


def find_all_possible_probabilities(like, all_con_prob_list):
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



def main(db, listem, i):
    c = sqlite3.connect(db).cursor()
    path = 'author_ground_truth' + str(i) + '.txt'
    file_to_save = open(path, 'w')
    for likepatterns in listem:
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
                liste.append(return_cardinality(pair, c, 450000))

            s = [str(k) for k in liste]
            file_to_save.write(newlike + ':' + ' '.join(s) + '\n')
        except:
            print(likepatterns)


def load_like_patterns(filename):
    list_of_patterns = []
    with open(filename, 'r') as file:
        for line in file:
            list_of_patterns.append(line.strip())
    return list_of_patterns


if __name__ == "__main__":
    
    db_path = 'path to database' #Path to an SQLite database where each dataset is a column.
    like_patterns_path = 'path to training dataset'
    file_to_save_ground = 'path to save ground truth'

    if not os.path.exists(like_patterns_path):
        print(f"Error: Like patterns file not found at {like_patterns_path}")
    else:
        list_of_patterns = load_like_patterns(like_patterns_path)
        datasetsize = 450000
        main(db_path, list_of_patterns, file_to_save_ground, datasetsize)



