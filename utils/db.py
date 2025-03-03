import chromadb
import pymysql
from pymysql.err import OperationalError
import dotenv

from typing import List

config = dotenv.dotenv_values(".env")

class MySQLConnection:
    def __init__(self):
        self.try_connect()
    
    def try_connect(self):
        host = config["host"]
        port = config["mysql_port"]
        user = config["mysql_user"]
        passwd = config["mysql_passwd"]
        db = config["mysql_db"]
        self.conn = pymysql.connect(host=host, port=int(port), user=user, passwd=passwd, db=db)
    
    def get_metadata(self, id):
        query = f"""
            SELECT A.id, image_url_small, atk, def, level, archetype, attribute, A.desc, frame_type, kor_desc, kor_name, name, race, type
            FROM card_model as A left join card_image as B on A.id=B.id where A.id={id};
        """
        cursor = self.execute_query(query)
        col_desc = cursor.description
        result = {}
        data = cursor.fetchone()
        try:
            for desc, value in zip(col_desc, data):
                key = desc[0]
                result[key] = value
            return result
        except:
            print(1)

    def execute_query(self, query, ttl=1):
        if ttl < 0:
            raise ConnectionError

        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            return cursor
        except OperationalError:
            self.try_connect()
            return self.execute_query(query, ttl - 1)

    
    def get_all_card_ids(self):
        query = f"SELECT id FROM card_model"
        cursor = self.execute_query(query)
        
        result = []
        for _ in range(cursor.rowcount):
            result.append(cursor.fetchone()[0])
        return result


class ChromaDB:
    def __init__(self):
        host = config["host"]
        port = config["chroma_port"]
        collection_name = config["chroma_collection"]
        self.client = chromadb.HttpClient(host, port)
        self.collection = self.client.get_or_create_collection(collection_name)
    
    def remove_none_values(self, metadata: List[dict]):
        return [self.remove_none_value(data) for data in metadata]
    
    def remove_none_value(self, metadata: dict):
        for key in metadata:
            if metadata[key] is None:
                metadata[key] = ""
        return metadata

    def update_one(self, id, embed, metadata: dict):
        metadata = self.remove_none_value(metadata)
        return self.collection.update(
            ids=[id],
            metadatas=[metadata],
            embeddings=[embed],
        )

    def insert_one(self, id, embed, metadata: dict):
        metadata = self.remove_none_value(metadata)
        return self.collection.add(
            ids=[id],
            metadatas=[metadata],
            embeddings=[embed],
        )
    
    def insert(self, ids, embeds, metadata: List[dict]):
        metadata = self.remove_none_values(metadata)
        return self.collection.add(
            ids=ids,
            metadatas=metadata,
            embeddings=embeds
        )
    
    def search_by_embed(self, embed, n_result=1):
        result = self.collection.query(query_embeddings=[embed], n_results=n_result)
        return result["metadatas"]