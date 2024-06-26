import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PdfSplitter:

    def __init__(self,
                 src = "./pdfs",
                 chunk_size = 300,
                 chunk_overlap = 15,):
        self.src = src
        pdf_splits = self.crawl_pdfs(src)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        self.splits = self.text_splitter.split_documents(pdf_splits)

    def crawl_pdfs(self, src):
        pdf_splits = []
        for root, dirs, files in os.walk(src):
            files = [file for file in files if file.endswith(".pdf")]
            for file in files:
                loader = PyPDFLoader(f"{root}/{file}")
                pdf_splits.extend(loader.load())
            dirs = [dir for dir in dirs if "." not in dir]
            for dir in dirs:
                temp = self.crawl_pdfs(f"{root}/{dir}")
                pdf_splits.extend(temp)
        return pdf_splits

    def get_splits(self):
        return self.splits