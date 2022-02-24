import sqlite3
from collections import Counter
import pandas as pd

query = 'select discipline from arxivmetadata where arxiv_id == "{}"'
db = "/home/magnetification/Documents/AI/Scriptie/Data/unarXive-2020/database/refs.db"

def createConnection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def checkField(file, query, arxiv_ids):
    ''' Check the research field of papers in csv
    '''
    db = createConnection(file)
    cur = db.cursor()
    queries = [query.format(id) for id in arxiv_ids]
    cursorlist = [cur.execute(q).fetchall() for q in queries]
    arx_disc = {arxiv_ids[i]:cursorlist[i] for i in range(len(cursorlist))}
    check = [(('cs',) in d) for d in arx_disc.values()]
    print(Counter(check))

    return arx_disc, cursorlist

arxiv_ids = list(pd.read_csv('testdata/ids.csv')['0'])
ad, cursorlist = checkField(db, query, arxiv_ids)

# Test connection to database
# conn = createConnection(db)
# cur = conn.cursor()
