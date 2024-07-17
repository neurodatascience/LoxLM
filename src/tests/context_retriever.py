from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_milvus.vectorstores import Milvus as m

from utils.bids_split import BidsSplitter
from utils.pdf_split import PdfSplitter
from utils.example_loader import ExampleLoader
import json

URI = "http://localhost:19530"

connection_args = {"uri": URI}

CONTEXT_COLLECTION = "context_db"

bids_splitter = BidsSplitter()
bids_splits = bids_splitter.get_splits()

pdf_splitter = PdfSplitter()
pdf_splits = pdf_splitter.get_splits()

all_context = bids_splits

print("Embedding Model Load")
# Embedding Model Initialization

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Context
print("Context Store")
context_store = m(
    embedding_function=hf,
    connection_args=connection_args,
    collection_name=CONTEXT_COLLECTION,
    drop_old=True,
).from_documents(
    all_context, embedding=hf, collection_name=CONTEXT_COLLECTION, connection_args=connection_args
)

examples_test, _ = ExampleLoader().get_splits()
inputs = [test['h'] for test in examples_test]


# Construct Prompt
print("Retriever")
retriever = context_store.as_retriever(search_kwargs={"k": 2, "fetch_k": 10})

def format_docs(d):
    return str(d)

inputs = inputs[:5]

context = [format_docs(retriever.invoke(input)) for input in inputs]

combined = []
for inputs, context in zip(inputs, context):
    dic = {}
    dic['input'], dic['context'] = inputs, context
    combined.append(dic)
with open("context_results.json",'w') as f:
    json.dump(combined,f,indent = 4)
