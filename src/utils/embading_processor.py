import chromadb
from chromadb.types import Collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from src.client.openai_client import OpenAiClient


class EmbeddingProcessorNonChroma:
    def __init__(self):
        self.embeddings_data = []

    def embedding(self, chunks: list[str]):
        client = OpenAiClient().create()
        if client is None:
            return None
        for text in chunks:
            resp = client.embeddings.create(input=text, model="text-embedding-3-small")
            print(resp.data[0].embedding)
            self.embeddings_data.append(resp.data[0].embedding)
        return None


class ChromaClient:
    def __init__(self, chroma_data_path: str):
        self.client = chromadb.PersistentClient(path=chroma_data_path)
        print(self.client.list_collections())

    # self.embedding_func = embedding_functions.OpenAIEmbeddingFunction(
    def create_collection(self, collection_name: str) -> Collection:
        try:
            collection = self.client.get_collection(name=collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except chromadb.errors.NotFoundError:
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small"),
                metadata={"hnsw:space": "cosine"}
            )
        return collection

    def add_collection(self, collection, documents: list[str]):
        if collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")
        collection.add(
            ids=[f"id{i}" for i in range(len(documents))],
            documents=documents
        )

    def execute_query(self, collection, query: str, n_results: int = 5):
        if collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")
        return collection.query(
            query_texts=[query],
            n_results=n_results
        )


if __name__ == '__main__':
    chroma_client = ChromaClient(chroma_data_path="./../../chroma/rag_db")
    collections = chroma_client.create_collection(collection_name="rag_collection1")
    chroma_client.add_collection(collections, documents=["hola", "que", "tal"])
