from __future__ import annotations

import getpass
import json
import os

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from utils.example_loader import ExampleLoader
from utils.multi_example_selector import Example
from utils.multi_example_selector import MultiExampleSelector
from utils.prompt_logger import PromptLogger

# from sentence_transformers import SentenceTransformer

os.environ["OPENAI_API_KEY"] = getpass.getpass()


# Fields to expand
tokenize_keys = ["SeriesDescription", "ProtocolName"]
# Instantiate ExampleLoader
store = ExampleLoader(file="./utils/examples_simple.json", tokenize=True, keys=tokenize_keys)
# Grab splits for testing and storage form ExampleLoader
examples_test, examples_store = store.get_splits(randomize=True)
# for loading actual dicoms
"""
_, examples_store = store.get_splits(randomize = True, test_split = 0)
dicoms = ExampleLoader(file = "./input_dicoms.json", tokenize = True, keys = tokenize_keys, test_split =1)
examples_test, _ = dicoms.get_splits()
"""

# Create embedding model
"""
model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
model = SentenceTransformer(model_name)
"""

model = OpenAIEmbeddings(model="text-embedding-3-small")


# Instantiate Example Selector
examples_selector = MultiExampleSelector(examples=examples_store, k=20, model=model)


# Load LLM
print("Load LLM")
llm = ChatOpenAI(temperature=0, model="gpt-4o")

print("Model Loaded")
# Messages Prompt for examples, takes all fields
example_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Series Description: {series_description}, Protocol"
            "Name: {protocol_name}, Task Name: {task_name},Repetition Time:"
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
            "Name: {protocol_name}, Task Name: {task_name} Repetition Time:"
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
parser = PydanticOutputParser(pydantic_object=Example)


# Constructs the final prompt
# TODO: Include BIDS context as well
final_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are an expert in DICOM to BIDs conversion."
            "You will be given a variety of DICOM fields and you will use"
            "these fields to return a valid BIDs suffix."
            "Return the input and suffix in json format."
            "{format_instructions}"
            "Please use the following examples to guide your response."
            "The only valid suffixes are bold, T1w, T2w, dwi. Do not "
            "return any other suffix."
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
prompt_logger = PromptLogger()


# The chain of commands that are invoked
@chain
def rag_chain(example):
    args = example
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
# examples_test = examples_test[:20]
# Dumps dictionary from the Example Pydantic Objects
examples = [example.model_dump() for example in examples_test]
# Removes and stores the ground truth suffix value
Y_true = [example.pop("suffix") for example in examples]
# Runs a batch of example inputs
print("batch chain")
model_outputs = rag_chain.batch(examples)
prompt_logger.write_logs()

# Creates a list of dictionaties of inputs and model outputs
combined = []
for y, output, input in zip(Y_true, model_outputs, examples):
    try:
        obj_dict = output.dict()
        obj_dict["actual"] = y
        obj_dict["input"] = input
        combined.append(obj_dict)
    except Exception:
        combined.append(f"{output}, actual: {y}, input: {input}")
# Writes outputs into a json file
with open("model_outputs_gpt4o.json", "w") as f:
    json.dump(combined, f, indent=4)
