import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownTextSplitter

class BidsSplitter:

    def __init__(self,
                 src = "./bids-specification/src",

                 ignore = [],
                 chunk_size = 300,
                 chunk_overlap = 15,
                 ):
        self.src = src
        self.md_splitter = MarkdownTextSplitter()
        md_splits = self.crawl_specification(src,ignore)
        self.splits = self.md_splitter.create_documents(md_splits)
  #      self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
  #      self.splits = self.text_splitter.split_documents(md_splits)



    def crawl_specification(self, src, ignore=[]):
        md_splits = []
        for root, dirs, files in os.walk(src):
            files = [file for file in files if file.endswith(".md")]
            for file in files:
                with open(f"{root}/{file}") as f:
                    text = f.read()
                    md_splits.extend(self.md_splitter.split_text(text))
            dirs = [dir for dir in dirs if "." not in dir]
            for dir in dirs:
                temp = self.crawl_specification(f"{root}/{dir}")
                md_splits.extend(temp)
        return md_splits

    def get_splits(self):
        return self.splits

    
