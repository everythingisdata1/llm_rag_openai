from dotenv import load_dotenv
from openai import OpenAI

from src.client.openai_client import OpenAiClient
from src.utils.chunk_processor import ChunkProcessor
from src.utils.document_reader import DocumentReader
from src.utils.embaddings_creator_chroma import EmbeddingsProcessor

load_dotenv()


class RagPipeline:
    def __init__(self, collection_name, embad_model, chroma_data_path, file_path):
        self.doc_reader = DocumentReader(file_path=file_path)
        self.chunks_processor = ChunkProcessor(chunk_size=1000)
        self.chroma_db_client = EmbeddingsProcessor(collection_name=collection_name, embed_model=embad_model,
                                                    chroma_data_path=chroma_data_path)
        self.llm_model: OpenAI = None

    def initialize_rag_pipeline(self):
        # Step 1: Read the document
        document_text = self.doc_reader.read_pdf()

        # Step 2: Chunk the document
        document_chunks = self.chunks_processor.chunk_data(document_text)

        # Step 3: Create and populate the ChromaDB collection
        self.chroma_db_client.create_collection()
        self.chroma_db_client.add_collection(document_chunks)

        # Step 4: Initialize the LLM model
        self.llm_model = OpenAiClient().create()

    def user_query(self, query_text, n_results=5):
        if self.llm_model is None:
            raise ValueError("RAG pipeline not initialized. Call initialize_rag_pipeline() first.")

        # Step 5: Retrieve relevant chunks from ChromaDB
        results = self.chroma_db_client.execute_query(query=query_text, n_results=n_results)
        relevant_chunks = results['documents'][0]

        # Step 6: Generate a response using the LLM model
        context = "\n".join(relevant_chunks)
        print("Relevant Chunks:", relevant_chunks)  # Debugging line
        prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
        print("Prompt to LLM:", prompt)  # Debugging line

        response = self.llm_model.completions.create(prompt=prompt, model="gpt-5")
        print("LLM Response:", response)  # Debugging line

        return response.choices[0].message.content


if __name__ == '__main__':
    rag_pipeline = RagPipeline(
        collection_name="rag_collection",
        embad_model="all-MiniLM-L6-v2",
        chroma_data_path="./../chroma/rag_db",
        file_path="./../data/test.txt"
    )
    rag_pipeline.initialize_rag_pipeline()
    answer = rag_pipeline.user_query("whis the functionally fo my function  doSearchProduct in my code")
    print(answer)
