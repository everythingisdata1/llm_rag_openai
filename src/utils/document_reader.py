import os.path

from PyPDF2 import PdfReader


class DocumentReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_pdf(self) -> str:
        ext = os.path.splitext(self.file_path)[1].lower()
        print(ext)
        if ext == ".pdf":
            reader = PdfReader(self.file_path)
            return "\n".join(page.extract_text() for page in reader.pages)
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return "".join(file.readlines())


if __name__ == "__main__":
    # dr = DocumentReader("./../../data/DsTree.pdf")
    # docs = dr.read_pdf()
    # print(docs)
    dr = DocumentReader("./../../data/test.txt")
    print(dr.read_pdf())
