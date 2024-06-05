import os
import json

from langchain_community.vectorstores import Milvus as m

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassThrough
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from utils.bids_split import BidsSplitter
from utils.pdf_split import PdfSPlitter
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

all_context = pdf_splits + bids_splits

with open("examples.json","r") as f:
    examples = json.load(f)
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
    emedding_function = hf,
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
)
#Context
print("Context Store")
context_store = m(
    emedding_function = hf,
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

example_prompt = PromptTemplate(
    input_variables = ["SeriesDescription", "ProtocolName", "index"]
    template = """
        SeriesDescription: "{SeriesDescription}" \n ProtocolName: "{ProtocolName}"\n Suffix: "{index}"
    """
)
print(example_prompt)
few_shot_prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt = example_prompt
    suffix = "SeriesDescription: {SeriesDescription} \n ProtocolName: {ProtocolName} \n what is the suffix?",
    input_variables = ["SeriesDescription", "ProtocalName"],
)
print(few_shot_prompt)
#Load LLM
print("Load LLM")
llm = Ollama(
    model = "gemma:2b"
    callback_manager = CallbackManager(
        [StreamingStdOutCallbackHandler()]
    ),
    stop = ["<|eot_id|>"],
)

#Construct Prompt
print("Retriever")
retriever = vector_store.as_retriever()
print("Rag Prompt")
rag _prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert in DICOM to BIDs conversion. You will be asked to provide the BIDs suffix for a given SeriesDescription and ProtocolName."),
        few_shot_prompt,
        ("human", "{Question}"),
    ]
)
print("Rag_chain")
rag_chain = (
    {"context": retriever, "Question": RunnablePassthrough()}
    | rag_prompt
    | llm
)
print(rag_chain.invoke("Series Description = dots_motion, ProtocalName = dots_motion"))
