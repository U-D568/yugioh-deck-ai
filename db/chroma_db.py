from typing import List

import chromadb
import dotenv


config = dotenv.dotenv_values(".env")


class ChromaDBConnection:
    def __init__(self):
        host = config["host"]
        port = config["chroma_port"]
        collection_name = config["chroma_collection"]
        self.client = chromadb.HttpClient(host, port)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

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
        return self.collection.add(ids=ids, metadatas=metadata, embeddings=embeds)

    def search_by_embed(self, embed, n_result=1):
        result = self.collection.query(query_embeddings=[embed], n_results=n_result)
        return result["metadatas"]
