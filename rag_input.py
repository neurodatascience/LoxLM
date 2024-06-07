import os
import json

from langchain_community.vectorstores import Milvus as m

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_core.tools import tool

from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate

from utils.bids_split import BidsSplitter
from utils.pdf_split import PdfSplitter
#Database Parameters
print("Database Parameters")
URI = 'http://localhost:19530'

connection_args = {'uri': URI}

CONTEXT_COLLECTION = "context_db"
EXAMPLE_COLLECTION = "example_db"
#Data Loading
print("Data Loading")
bids_splitter = BidsSplitter()
bids_splits = bids_splitter.get_splits()

pdf_splitter = PdfSplitter()
pdf_splits = pdf_splitter.get_splits()

all_context = bids_splits
examples = []
with open("examples_clean.json","r") as f:
    examples2 = json.load(f)
    for example in examples2:
        series_description = example["SeriesDescription"]
        protocol_name = example["ProtocolName"]
        suffix = example["index"]
        formatted = {"h": f"SeriesDescription: {series_description}\nProtocolName: {protocol_name}","bot":f"Suffix: {suffix}"}
        examples.append(formatted)

print("Embedding Model Load")
#Embedding Model Initialization

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
print("Example Vector Store")
#Vector Store Initialization
#Examples
example_store = m(
    embedding_function = hf,
    connection_args = connection_args,
    collection_name = EXAMPLE_COLLECTION,
    drop_old = True,
)
print("Example Selector")
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    hf,
    example_store,
    k=5,
    input_keys =["h"],
)
#Context
print("Context Store")
context_store = m(
    embedding_function = hf,
    connection_args = connection_args,
    collection_name = CONTEXT_COLLECTION,
    drop_old = True,
).from_documents(
    all_context,
    embedding = hf,
    collection_name = CONTEXT_COLLECTION,
    connection_args = connection_args
)

#ExamplePrompt

example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{h}"),
                ("ai", "{bot}"),
                ] 

        )
print(example_prompt)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector = example_selector,
    example_prompt = example_prompt,
    input_variables = ["h"],
)
print(few_shot_prompt)
#Load LLM
print("Load LLM")
llm = Ollama(
    model = "llama3",
    callback_manager = CallbackManager(
        [StreamingStdOutCallbackHandler()]
    ),
    stop = ["<|eot_id|>"],
    temperature = 0.05,
    top_k = 30,
    top_p = 0.5,
)

#Construct Prompt
print("Retriever")
retriever = context_store.as_retriever(search_kwargs={'k':2,'fetch_k':10})
print("Rag Prompt")


final_prompt = ChatPromptTemplate.from_messages(
        [
        SystemMessagePromptTemplate.from_template("You are an expert in DICOM to BIDs conversion. You will be asked to provide the BIDs suffix for a given SeriesDescription and ProtocolName. Use the following context from the bids specification to aid your answer: {context}\n Return a suffix from the following list (bold, T1w, T2w, dwi). Use the following examples to understand what to return and to use as reference about what to return."),
        few_shot_prompt,
        HumanMessagePromptTemplate.from_template("{h}\n Suffix:"),
        ]
    )

def format_docs(d):
    return str(d)

rag_chain = (
    {   "context": format_docs | retriever,
        "h": RunnablePassthrough(),
     }
    | final_prompt
    | llm
)

def fields_to_string(fields: list[dict]):
    return [(f"SeriesDescription: {field["SeriesDescription"]}\nProtocolName: {field["ProtocolName"]}",f"Suffix: {field["index"]}") for field in fields]

with open("examples_test.json", "r") as f:
    testers = json.load(f)
testers = fields_to_string(testers)
inputs = [test[0] for test in testers]
outputs = [test[1] for test in testers]
model_outputs = rag_chain.batch(inputs = inputs)

outs = dict(zip(outputs,model_outputs))

with open("model_outputs.json", "w") as f:
    json.dump(outs,f)
        


