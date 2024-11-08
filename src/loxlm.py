from __future__ import annotations

import getpass
import json
import os
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.runnables import chain
from langchain_milvus.vectorstores import Milvus as m
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from utils.bids_split import BidsSplitter
from utils.dicom_loader import DicomLoader
from utils.example_loader import ExampleLoader
from utils.multi_example_selector import Example
from utils.multi_example_selector import MultiExampleSelector
from utils.prompt_logger import PromptLogger


num_examples : int = 30
tokenize = True
tokenize_keys : list[str] = ["SeriesDescription", "ProtocolName"]
examples_file = "./utils/examples_simple.json"
openai_embedding_model = "text-embedding-3-small"
local_embedding_model = "sentence-transformers/distiluse-base-multilingual-cased-v2"

weights = {
    "series_description": 0.6,
    "protocol_name": 0.6,
    "task_name": 0.2,
    "repetition_time": 0.2,
    "echo_time": 0.2,
    "inversion_time": 0.2,
    "pulse_sequence_type": 0.3,
    "flip_angle": 0.2,
    "manufacturer": 0.4,
    "model": 0.2,
}
openai_model ='gpt-4o'
local_model = "gemma2"

URI = "http://localhost:19530"

connection_args = {"uri": URI}

CONTEXT_COLLECTION = "context_db"

huggingface_model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}


class LoxLM:
    def __init__(
            self,
            input_dicoms: str | None = None,
            test: bool = True,
            output_file: str = './output.json',
            log_prompts: bool = False,
            context: bool = False,
            openai: bool = True,
        ):
        """
        Object to process dicoms to generate bids suffixes.

        Parameters:
        input_dicoms: str | None - String of address of dicom dataset to ingest.
        Does not need to be inputted here. Can also be inputted using batch_input method.

        test: bool - Whether or not to run a test instance. This splits the examples dataset
        to test on a portion of these and using the rest for the examples store.

        output_file: str - The address where the output json will be written

        log_promtps: bool - Whether or not to print a text file of all prompts sent generated.
        Useful for testing.

        context: bool - Whether or not to create a context store from bids specification. If this
        is set to True make sure to spin up Milvus docker containers using 'docker compose up -d'.
        It is recommended to not use this.

        openai: bool - Whether or not to use openai models. If False local models will be used. This assumes
        that an ollama instance is running locally with a specified model pulled. It is recommended to use
        openai by default. The default openai model is 'gpt-4o'
        """
        self.context = context
        self.output_file = output_file
        if input_dicoms and test:
            raise ValueError("Can't have input file and test True")

        example_loader = ExampleLoader(file = examples_file, tokenize = tokenize, keys = tokenize_keys)

        if input_dicoms:
            self.input_dict = DicomLoader(input_dicoms).git_dict()
            _, self.examples_store = example_loader.get_splits(test_split = 0)
        else:
            examples_test, self.examples_store = example_loader.get_splits(randomize=True)
            self.input_dict = [example.model_dump() for example in examples_test]
        if context:
            #something to test if a docker container is running
            bids_splitter = BidsSplitter()
            bids_splits = bids_splitter.get_splits()
        if openai:
            os.environ["OPENAI_API_KEY"] = getpass.getpass()
            self.embedding_model = OpenAIEmbeddings(model = openai_embedding_model)
            self.chat_model = ChatOpenAI(temperature = 0, model = openai_model)
            if context:
                self.context_store = m(
                                    embedding_function=embedding_model,
                                    connection_args=connection_args,
                                    collection_name=CONTEXT_COLLECTION,
                                    drop_old=True,
                                ).from_documents(
                                    bids_splits,
                                    embedding=embedding_model,
                                    collection_name=CONTEXT_COLLECTION,
                                    connection_args=connection_args,
                                )
                self.retriever = context_store.as_retriever(search_kwargs={"k": 3, "fetch_k": 10})

        else:
            self.embedding_model = SentenceTransformer(local_embedding_model)
            self.chat_model = Ollama(
                                        model="gemma2",
                                        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                                        stop=["<start_of_turn>", "<end_of_turn>"],
                                        temperature=0.00,
                                        top_k=15,
                                        top_p=0.2,
                                    )
            if context:
                context_embedder = HuggingFaceBgeEmbeddings(model_name=huggingface_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
                self.context_store = m(
                    embedding_function=context_embedder,
                    connection_args=connection_args,
                    collection_name=CONTEXT_COLLECTION,
                    drop_old=True,
                ).from_documents(
                    bids_splits,
                    embedding=context_embedder,
                    collection_name=CONTEXT_COLLECTION,
                    connection_args=connection_args,
                )
                self.retriever = context_store.as_retriever(search_kwargs={"k": 3, "fetch_k": 10})
        self.example_selector = MultiExampleSelector(examples=self.examples_store, k=12, model=self.embedding_model, weights = weights)
        self.parser = PydanticOutputParser(pydantic_object=Example)
        self.prompt = self.make_prompt(context, self.parser)

        if log_prompts:
            self.prompt_logger = PromptLogger(file= './prompt_logs.txt')




    def make_prompt(self, context, parser):
        """Return prompt with or without context."""
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
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_selector=self.example_selector,
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
        if context:
            final_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        "You are an expert in DICOM to BIDs conversion."
                        "Use this context as your background for BIDS: {context}"
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
        else:
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
        final_prompt = final_prompt.partial(
            format_instructions=parser.get_format_instructions(),
        )
        return final_prompt



    def expand(value: str):
        """Expands strings to make more embeddable."""
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
        """Writes a dict to a string."""
        string = ""
        for key, value in dic.items():
            if key in tokenize_keys:
                value = expand(value)
            string += f"{key}: {value}"
        return string

    def grab_sources(self, input_dicts):
        """Separates file and root from dictionary."""
        sources = []
        for dic in input_dicts:
            file = dic.pop('file', None)
            root = dic.pop('root', None)
            source = f"{root}/{file}"
            sources.append(source)
        return sources, input_dicts

    def grab_groundtruth(self, input_dicts):
        """Separates ground truth suffix from dictionary."""
        ground_truths =[]
        for dic in input_dicts:
            ground_truth = dic.pop('suffix')
            ground_truths.append(ground_truth)
        return ground_truths, input_dicts

    def output_mappings(self,input_dict, model_outputs, sources, ground_truths,  output_file,):
        """Writes outputs and inputs into a json file."""
        if not sources:
            sources = [""]*len(input_dict)
        if not ground_truths:
            ground_truths = [" "]*len(input_dict)
        results = []
        for input, output, source, truth in zip(input_dict, model_outputs, sources, ground_truths):
            try:
                obj_dict = output.dict()
                obj_dict['input'] = input
                obj_dict['actual'] = truth
                obj_dict['source'] = source
                results.append(obj_dict)
            except Exception as e:
                print(e)
                results.append(f"{output},input:{input},actual:{truth},source: {source}")
        if not output_file:
            output_file = self.output_file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent = 4)


    def batch(self, input_dict, max_workers = 16):
        """
        Invokes prompts chain in parallel.

        Adjust 'max_workerrs' depending on your machines resources.
        """
        outputs = []
        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            futures = [executor.submit(self.invoke, inp) for inp in input_dict]
            for future in as_completed(futures):
                output = future.result()
                outputs.append(output)
        return outputs

    def invoke(self, input_dict):
        """Chain of input to prompt to LLM to parsing."""
        if self.context:
            string = dict_to_string(input_dict)
            collected_context = self.retriever.invoke(string)
            input_dict['context'] = collected_context
        prompt = self.prompt.invoke(input_dict)
        if self.prompt_logger:
            self.prompt_logger.add_prompt(prompt, input = input_dict)
        output = self.chat_model.invoke(prompt)
        try:
            parsed_output = self.parser.invoke(output)
        except Exception as e:
            print(e)
            parsed_output = output
        return parsed_output

    def batch_input(self, input_dicoms: str | None = None, output_file: str | None = None):
        """Used to invoke LoxLM object.

        Parameters:

        input_dicoms: str | None - The address where to find dicom directory. Can be None if
        LoxLM is in test mode or if a directory was inputted in LoxLM instantiation.

        output_file: str | None - The address of desired output directory for the batch.
        """
        if not input_dicoms and not self.input_dict:
            raise ValueError("No input data.")
        if not input_dicoms:
            if 'file' in self.input_dict[0]:
                sources, input_dict = self.grab_sources(self.input_dict)
                ground_truths = None
            else:
                ground_truths, input_dict = self.grab_groundtruth(self.input_dict)
                sources = None
        if input_dicoms:
            input_dict = DicomLoader(input_dicoms).get_dict()
            sources, input_dict = self.grab_sources(input_dict)
            ground_truths = None
        model_outputs = self.batch(input_dict = input_dict)
        self.output_mappings(input_dict, model_outputs, sources, ground_truths, output_file )
