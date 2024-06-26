import os
import re

import nltk
from langchain_text_splitters import (
    MarkdownTextSplitter,
)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

nltk.download('punkt')
class BidsSplitter:

    def __init__(self,
                 src = "../bids-specification/src",

                 ignore = ['README.md','CHANGES.md','extensions.md','licenses.md',
                     'contributors.md'],
                 chunk_size = 120,
                 chunk_overlap = 15,
                 ):
        self.src = src
        self.md_splitter = MarkdownTextSplitter()
        md_splits = self.crawl_specification(src,ignore)

        self.splits = self.md_splitter.create_documents(md_splits)
        #self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        #self.splits = self.text_splitter.split_documents(md_splits)



    def crawl_specification(self, src, ignore=[]):
        md_splits = []
        for root, dirs, files in os.walk(src):
            files = [file for file in files if (file.endswith(".md") and file
                not in ignore)]
            print(files)
            for file in files:
                with open(f"{root}/{file}") as f:
                    text = f.read()
                    text = self.clean_text(text)
                    md_splits.extend(self.md_splitter.split_text(text))
            dirs = [dir for dir in dirs if "." not in dir]
            for dir in dirs:
                temp = self.crawl_specification(f"{root}/{dir}")
                md_splits.extend(temp)
        return md_splits

    def clean_text(self, text):
        tokens = word_tokenize(text)
        #remove non word or space characters
        cleaned_tokens = [re.sub(r'[^\w]','', token) for token in tokens]
        cleaned_tokens = [token.lower() for token in cleaned_tokens]
        #remove stop words
        stop_words = set(stopwords.words('english'))
        cleaned_tokens = [token for token in cleaned_tokens if token not in stop_words]
        cleaned_text = " ".join(cleaned_tokens)
        return cleaned_text

    def get_splits(self):
        return self.splits


