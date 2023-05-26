import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from collections import OrderedDict
import itertools


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