from __future__ import annotations

import json

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.runnables import chain
from langchain_milvus.vectorstores import Milvus as m
from sentence_transformers import SentenceTransformer
from utils.bids_split import BidsSplitter
from utils.example_loader import ExampleLoader
from utils.multi_example_selector import Example
from utils.multi_example_selector import MultiExampleSelector
from utils.pdf_split import PdfSplitter
from utils.prompt_logger import PromptLogger

# Fields to expand
tokenize_keys = ["SeriesDescription", "ProtocolName"]
# Instantiate ExampleLoader
el = ExampleLoader(file="./utils/examples.json", tokenize=True, keys=tokenize_keys)
# Grab splits for testing and storage form ExampleLoader
examples_test, examples_store = el.get_splits(randomize=True)


# Create embedding model
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
model = SentenceTransformer(model_name)

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# Instantiate Example Selector
examples_selector = MultiExampleSelector(examples=examples_store, k=5, model=model)

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


# Load LLM
print("Load LLM")
llm = Ollama(
    model="gemma2",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    stop=["<start_of_turn>", "<end_of_turn>"],
    temperature=0.00,
    top_k=15,
    top_p=0.2,
)
print("Model Loaded")
# Context
print("Context Store")
context_store = m(
    embedding_function=hf,
    connection_args=connection_args,
    collection_name=CONTEXT_COLLECTION,
    drop_old=True,
).from_documents(
    all_context,
    embedding=hf,
    collection_name=CONTEXT_COLLECTION,
    connection_args=connection_args,
)

print("Retriever")
retriever = context_store.as_retriever(search_kwargs={"k": 3, "fetch_k": 10})
# Messages Prompt for examples, takes all fields
example_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Series Description: {series_description}, Protocol"
            "Name: {protocol_name}, Task Name: {task_name}, Repetition Time:"
            "{repetition_time}, Echo Time: {echo_time}, Inversion Time:"
            "{inversion_time}, Pulse Sequence Type: {pulse_sequence_type}, Flip"
            "Angle: {flip_angle}, Manufacturer: {manufacturer}, Model:{model}",
        ),
        ("ai", "Suffix: {suffix}"),
    ],
)
# Messages prompt for input, there is no suffix inputted
example_prompt_no_suffix = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Series Description: {series_description}, Protocol"
            "Name: {protocol_name}, Task Name: {task_name}, Repetition Time:"
            "{repetition_time}, Echo Time: {echo_time}, Inversion Time:"
            "{inversion_time}, Pulse Sequence Type: {pulse_sequence_type}, Flip"
            "Angle: {flip_angle}, Manufacturer: {manufacturer}, Model:{model}",
        ),
        ("ai", "Suffix: "),
    ],
)


# Creates frew shot example prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=examples_selector,
    example_prompt=example_prompt,
    input_variables=[
        "series_description",
        "protocol_name",
        "task_name",
        "repetition_time",
        "echo_time",
        "inversion_time",
        "pulse_sequence_type",
        "flip_angle",
        "manufacturer",
        "model",
    ],
)

# Creates pydantic output parser with template object being an Example
# TODO: Wrap this class in something to handle parsing errors

parser = PydanticOutputParser(pydantic_object=Example)


# Constructs the final prompt
# TODO: Include BIDS context as well
final_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are an expert in DICOM to BIDs conversion."
            "Use this context as your background for BIDS: {context}"
            "You will be given a variety of DICOM fields and you will use"
            "these fields to return a valid BIDs suffix."
            "Return the input and suffix in json format."
            "Do not explain your reasoning"
            "{format_instructions}"
            "Please use the following examples to guide your response."
        ),
        few_shot_prompt,
        example_prompt_no_suffix,
    ]
)

# Adds pydantic format instructions to prompt
final_prompt = final_prompt.partial(
    format_instructions=parser.get_format_instructions(),
)

# Creates Prompt Logger which records each prompt into a text file
prompt_logger = PromptLogger(file="./context_prompt_logs.txt")


def format_docs(d):
    return str(d)


def expand(value: str):
    tokens = []
    current_token = ""
    for char in value:
        if char.isupper():
            if current_token:
                if current_token[-1].isupper() or current_token[-1].isdigit():
                    current_token += char
                    continue
                else:
                    tokens.append(current_token)
            current_token = char
        elif char == "_":
            if current_token:
                tokens.append(current_token)
            current_token = ""
        elif char == "-":
            if current_token:
                tokens.append(current_token)
            current_token = ""
        elif char == " ":
            if current_token:
                tokens.append(current_token)
            current_token = ""
        else:
            current_token += char
    if current_token:
        tokens.append(current_token)
    return tokens


def dict_to_string(dic: dict, keys=tokenize_keys):
    string = ""
    for key, value in dic.items():
        if key in tokenize_keys:
            value = expand(value)
        string += f"{key}: {value}"
    return string


# The chain of commands that are invoked
@chain
def rag_chain(example):
    args = example
    string = dict_to_string(args)
    context = retriever.invoke(string)
    args["context"] = context
    # creates prompt from input arguments
    prompt = final_prompt.invoke(args)
    # Stores the resulting prompt
    prompt_logger.add_prompt(prompt, input=args)
    # Passes prompt to the LLM
    output = llm.invoke(prompt)
    # Parser parses llm output. Currently crashes annoyingly
    try:
        parsed_output = parser.invoke(output)
    except Exception:
        parsed_output = output
    return parsed_output


# prompt logger writes saved prompts into a text file

# Takes subset of test examples
# examples_test = examples_test[:5]
# Dumps dictionary from the Example Pydantic Objects
examples = [example.model_dump() for example in examples_test]
# Removes and stores the ground truth suffix value
Y_true = [example.pop("suffix") for example in examples]
# Runs a batch of example inputs
model_outputs = rag_chain.batch(examples)
prompt_logger.write_logs()

# Creates a list of dictionaties of inputs and model outputs
combined = []
for y, output, dicom in zip(Y_true, model_outputs, examples):
    if "context" in dicom:
        dicom.pop("context")
    try:
        obj_dict = output.model_dump()
        obj_dict["actual"] = y
        obj_dict["input"] = dicom
        combined.append(obj_dict)
    except Exception:
        combined.append({"output": output, "actual": y, "input": dicom})
# Writes outputs into a json file
with open("context_model_outputs.json", "w") as f:
    try:
        print(type(combined[0]))
        json.dump(combined, f, indent=4)
    except Exception:
        for dic in combined:
            f.write(dic)
