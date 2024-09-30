import sqlite3

def create_db(dataset, db_name):
    """
       Creates an SQLite database with a table named 'column' and populates it with data from a CSV file.

       Args:
           dataset (str): The file path to the CSV dataset. Each line in the dataset represents a row to be inserted.
           db_name (str): The name of the SQLite database to be created or modified.
       """
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    c.execute('''CREATE TABLE column
             (my_var2 INT, name text)''')
    count = 0
    lines = list()
    for line in open (dataset):
        line = line.strip()
        lines.append((count , line))
        count += 1

    c.executemany('INSERT INTO column VALUES (?,?)', lines)
    conn.commit()

#create_db('Datasets/author_name.csv', 'author.db')