# Please open the attached [Jupyter Notebook](similarity_search_duckdb.ipynb) in Google Colab and run it step by step.


# Install dependencies

``` python
pip install duckdb==0.8.0
```

``` python
pip install -U sentence-transformers
```

``` python
import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from collections import OrderedDict
import itertools
```

# similaritySearch class

- The ***similaritySearch*** class by default creates a table called "VECTORS" where all records from all tables are stored as vectors. This happens within the ***__create_vector_storage*** and ***insert_vector*** methods.

- There is a method called ***__to_vector*** that converts each incoming record from any table into a vector. Here, a pretrained model is used to have universal vectors that do not depend on the corpus of other incoming records. Techniques like ***word2vec***, ***tfidf***, ***bag of words**, and others would not be suitable as they require knowledge of all incoming records to generate a vector.

- The ***search_similarities*** method is not efficient because it only uses duckdb as a storage for vectors. The similarity computation is not done within the duckdb engine. Instead, all vectors are extracted and compared locally to measure the distance between the input row and existing vectors. It was not possible for me to create a scalar function to offload this computation to the engine.

- The cosine metric is used within the ***__cosine_distance_similarity*** method. The decision to use this metric is based on its good results for many types of cases and its insensitivity to vector length. The similarity between two words decreases as the distance between their vectors increases, and vice versa. Therefore, a value close to 0 indicates a good similarity between the input row and the current vector being compared.

- The remaining methods handle the logic for inserting tables and inserting data into existing tables.

``` python

class similaritySearch:
                
    def __init__(self, database_name):
        self.database_name = database_name
        self.con = duckdb.connect(database_name)
        self.__create_vector_storage()

    def __execute_commit(self, statement):        
        self.con.execute(statement)
        self.con.commit()        

    def __create_vector_storage(self, ):
        statement = '''CREATE TABLE IF NOT EXISTS VECTORS (id INTEGER, tbl VARCHAR, vector DOUBLE[])'''
        self.__execute_commit(statement)

    def create_table(self, schema):
        metadata = ', '.join([k + ' ' + v for k, v in schema['columns'].items()])
        statement = "CREATE TABLE IF NOT EXISTS {} ({})".format(schema['table_name'],metadata)
        self.__execute_commit(statement)


    def insert_data(self, table_name,  schema):
        columns = ', '.join(schema.keys())
        holders = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in schema.values()])
        statement = f"INSERT INTO {table_name} ({columns}) VALUES ({holders})"
        self.__execute_commit(statement)
        vector = self.__to_vector(holders)
        return vector
        

    def insert_vector(self, id, table_name, vector):
        statement = f"INSERT INTO VECTORS (id, tbl, vector) VALUES ({id}, '{table_name}', {vector})"
        self.__execute_commit(statement)


    def __to_vector(self, data):
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu', cache_folder='/')
        embeddings = model.encode(data, convert_to_tensor=True, normalize_embeddings=True)                
        return embeddings.tolist()

    def __cosine_distance_similarity(self, u, v):
        distance = 1.0 - (np.dot(u, v)/
                          (np.sqrt(sum(np.square(u))) * 
                           np.sqrt(sum(np.square(v))))
                          )
        return distance


    def __value(self, item):
        return item[1]    

    def search_similarities(self, query, table_name, top_k):

        sd = OrderedDict()

        holders = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in query.values()])
        query_vector = self.__to_vector(holders)

        rows =  self.con.execute(f"SELECT id, vector FROM VECTORS WHERE tbl = '{table_name}'")
    
        for row in rows.fetchall():            
            d = self.__cosine_distance_similarity(query_vector, row[1])            
            sd[row[0]] = d

        sd = OrderedDict(sorted(sd.items(), key=self.__value))

        return list(itertools.islice(sd.items(), top_k))
```
# DB is an object of the class similaritySearch

``` python
db = similaritySearch(':memory:')
```

# The dictionary ***schema_tables*** contains the schema of all the tables to be created.

``` python
schema_tables = [{
    "table_name": "MyTable",
    "columns": {
        "id": 'INTEGER',
        "column1": "VARCHAR",
        "column2": "VARCHAR",
        "column3": "VARCHAR",
        "column4": "INTEGER"
    }
}]
```

# We iterate through each element and create the tables.

``` python
for schema in schema_tables:
    db.create_table(schema)
    print(f"Table {schema['table_name']} created")
```    

# The dictionary ***data_tables*** contains the data to be added to each table.

``` python
data_tables = [{
"table_name": "MyTable",
    "data":{
            "id":5,
            "column1":"Random text for testing - movies",
            "column2":"movies",
            "column3":"Run Lola Run",
            "column4": 112
        }
    },

{
"table_name": "MyTable",
    "data":{
            "id":6,
            "column1":"Random text for testing - food",
            "column2":"food",
            "column3":"Bratwurst",
            "column4": 115
        }
    },
{
"table_name": "MyTable",
    "data":{
            "id":7,
            "column1":"Random text for testing - place",
            "column2":"place",
            "column3":"Neuschwanstein Castle",
            "column4": 123
        }
    },
{
"table_name": "MyTable",
    "data":{
            "id":8,
            "column1":"skjdlkajsasdf",
            "column2":"dasñldk´ñalskd",
            "column3":"dasd",
            "column4": 580
        }
    }, 
{
"table_name": "MyTable",
    "data":{
            "id":9,
            "column1":"aaaaaaaaaaaaaa",
            "column2":"33333333errrrrrrrrrr",
            "column3":"fgggggggggggg",
            "column4": 111
        }
    }                
]

```

# We iterate and insert the data into each table while simultaneously creating a vector for each added record.

``` python
for data in data_tables:
    vector = db.insert_data(data['table_name'], data['data'])
    print(f"record inserted in table {data['table_name']}")
    db.insert_vector(data['data']['id'],data['table_name'], vector )
    print(f"vector inserted")
```

# Here I'm simply requesting to display the 3 most similar vectors to an query or input vector, and clearly the results look quite good. The example below shows that the 3 most similar records are the records with id 8, 9, and 7.


``` python

query = {"column1": "skjdlkajsasdf", "column2": "dasñldk´ñalskd", "column3": "dasd", "column4": 580}
similarities = db.search_similarities(query, "MyTable", top_k=3)
similarities
```