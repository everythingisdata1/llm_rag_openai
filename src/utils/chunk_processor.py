from typing import Any

from src.utils.document_reader import DocumentReader


class ChunkProcessor:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def chunk_process(self, data):
        """Process the input data in chunks of specified size."""
        for i in range(0, len(data), self.chunk_size):
            yield data[i:i + self.chunk_size]


    def chunk_data(self, docs) -> list[Any]:
        return [chunk for chunk in self.chunk_process(docs)]


if __name__ == "__main__":
    dr = DocumentReader("./../../data/boe-2025.pdf")
    docs = dr.read_pdf()
    cp = ChunkProcessor(chunk_size=1000)
    documents=cp.chunk_data(docs)
    # print(len(documents))
