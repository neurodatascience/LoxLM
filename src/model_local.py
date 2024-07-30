from __future__ import annotations

import json

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.runnables import chain
from sentence_transformers import SentenceTransformer
from utils.example_loader import ExampleLoader
from utils.multi_example_selector import Example
from utils.multi_example_selector import MultiExampleSelector
from utils.prompt_logger import PromptLogger


# Fields to expand
tokenize_keys = ["SeriesDescription", "ProtocolName"]
# Instantiate ExampleLoader
el = ExampleLoader(file="./utils/examples.json", tokenize=True, keys=tokenize_keys)
# Grab splits for testing and storage form ExampleLoader
examples_test, examples_store = el.get_splits(randomize=True)


# Create embedding model
model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
model = SentenceTransformer(model_name)


# Instantiate Example Selector
examples_selector = MultiExampleSelector(examples=examples_store, k=50, model=model)


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
            "You will be given a variety of DICOM fields and you will use"
            "these fields to return a valid BIDs suffix."
            "Return the input and suffix in json format."
            "Only choose from the following suffixes: T1w, T2w, dwi, bold"
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
examples_test = examples_test[:50]
# Dumps dictionary from the Example Pydantic Objects
examples = [example.model_dump() for example in examples_test]
# Removes and stores the ground truth suffix value
Y_true = [example.pop("suffix") for example in examples]
# Runs a batch of example inputs
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
with open("model_outputs.json", "w") as f:
    json.dump(combined, f, indent=4)
