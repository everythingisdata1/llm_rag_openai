import chromadb
from chromadb.utils import embedding_functions

from src.utils.chunk_processor import ChunkProcessor
from src.utils.document_reader import DocumentReader


class EmbeddingsProcessor:
    def __init__(self, collection_name: str, embed_model: str, chroma_data_path: str):
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.chroma_data_path = chroma_data_path

        self.client = chromadb.PersistentClient(path=self.chroma_data_path)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embed_model)
        self.collection = None

    def create_collection(self):
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Collection '{self.collection_name}' already exists.")
        except chromadb.errors.NotFoundError:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_func,
                metadata={"hnsw:space": "cosine"}
            )

    def add_collection(self, documents: list[str]):
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")
        self.collection.add(
            ids=[f"id{i}" for i in range(len(documents))],
            documents=documents
        )

    def execute_query(self, query: str, n_results: int = 5):
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )


if __name__ == '__main__':
    ep = EmbeddingsProcessor(
        collection_name="rag_collection",
        embed_model="all-MiniLM-L6-v2",
        chroma_data_path="./../../chroma/rag_db"
    )
    documentReader = DocumentReader("./../../data/boe-2025.pdf")
    docs = documentReader.read_pdf()
    cp = ChunkProcessor(chunk_size=1000)
    # print(cp.chunk_data(docs))
    documents = cp.chunk_data(docs)

    ep.create_collection()

    ep.add_collection(
        documents=documents
    )
    print("Collection created and documents added.")
    results = ep.execute_query("please help me with the name of Chief Cashier of BOE  ", n_results=2)
    print("Query executed. Results:")
    print(results.keys())
    print(results['documents'])
