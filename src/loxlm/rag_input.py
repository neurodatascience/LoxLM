from __future__ import annotations

import json

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.tools.retriever import create_retriever_tool
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus.vectorstores import Milvus as m
from utils.bids_split import BidsSplitter
from utils.example_loader import ExampleLoader
from utils.pdf_split import PdfSplitter

# Database Parameters
print("Database Parameters")
URI = "http://localhost:19530"

connection_args = {"uri": URI}

CONTEXT_COLLECTION = "context_db"
EXAMPLE_COLLECTION = "example_db"
# Data Loading
print("Data Loading")
bids_splitter = BidsSplitter()
bids_splits = bids_splitter.get_splits()

pdf_splitter = PdfSplitter()
pdf_splits = pdf_splitter.get_splits()

all_context = bids_splits
#Load Examples
examples_test, examples_store = ExampleLoader().get_splits()

print("Embedding Model Load")
# Embedding Model Initialization

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
print("Example Vector Store")
# Vector Store Initialization
# Examples
example_store = m(
    embedding_function=hf,
    connection_args=connection_args,
    collection_name=EXAMPLE_COLLECTION,
    drop_old=True,
)
print("Example Selector")
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples_store,
    hf,
    example_store,
    k=5,
    input_keys=["h"],
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


class MRI_Schema(BaseModel):
    """Schema for MRI data."""

    series_description: str = Field(default="Not outputted",description="The series description inputted by the user.")
    protocol_name: str = Field(default = "not outputted",description="The protocol name inputted by the user.")
    bot: str | list = Field(default = "not outputted", description="BIDs suffix corresponding to the user's inputted DICOM fields.")


"""
    @validator('h')
    def complete_dicom_input(cls, field):
        pattern = r'SeriesDescription: (.+?)(?:\n)?ProtocolName: (.+?)$'
        match =  re.search(pattern, h)
        if match:
            series_description = match.group(1)
            protocol_name = match.group(2)
            return field
        else:
            raise ValueError("Improperly Formed Input")
"""
# ExamplePrompt

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{h}"),
        ("ai", "{bot}"),
    ]
)
print(example_prompt)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    input_variables=["h"],
)
print(few_shot_prompt)
# Load LLM
print("Load LLM")
llm = Ollama(
    model="gemma",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    stop=["<|eot_id|>"],
    temperature=0.00,
    top_k=15,
    top_p=0.2,
)

# Construct Prompt
print("Retriever")
retriever = context_store.as_retriever(search_kwargs={"k": 2, "fetch_k": 10})
retriever_tool = create_retriever_tool(
    retriever,
    "bids_specification_search",
    """Search for information about the BIDS specification.
    Use this to get context for DICOM to BIDs conversion.""",
)
print("Rag Prompt")

parser = PydanticOutputParser(pydantic_object=MRI_Schema)

final_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are an expert in DICOM to BIDs conversion."
            "You will be asked to provide the BIDs suffix for a given SeriesDescription and ProtocolName."
            "Use the following context from the bids specification to aid your answer: {context}"
            "\n Return a suffix from the BIDs specificatoin."
            "Use the following examples to understand what to return.\n{format_instructions}\n"
        ),
        few_shot_prompt,
        HumanMessagePromptTemplate.from_template("{h}\n Suffix:"),
    ]
)
final_prompt = final_prompt.partial(
    format_instructions=parser.get_format_instructions(),
)
"""
tools = [retriever_tool]
agent = create_structured_chat_agent(llm, retriever_tool, final_prompt, parser)

agent_executor = AgentExecutor(agent, tools, handle_parsing_errors=True)
"""
def format_docs(d):
    return str(d)

rag_chain = (
    {

        "context": format_docs | retriever,
        "h": RunnablePassthrough(),

     }
    | final_prompt
    | llm
    | parser
)



#examples_test = examples_test[:20]
inputs = [test['h'] for test in examples_test]
outputs = [test['bot'] for test in examples_test]
inputs = inputs[200:400]
model_outputs = rag_chain.batch(inputs)
"""
for input in inputs:
    try:
        out = rag_chain.invoke(input)
        model_outputs.append(out)
    except:
        print(f"failed to return properly. Input: {input}")
"""
print(model_outputs)
print(outputs)
combined = []
for obj, x, y in zip(model_outputs, inputs, outputs):
    obj_dict = obj.dict()
    obj_dict["actual"] = y
    obj_dict["input"] = x
    combined.append(obj_dict)

with open("model_outputs_2.json", "w") as f:
    json.dump(combined, f, indent=4)
