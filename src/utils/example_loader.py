from .multi_example_selector import Example
from pathlib import Path
import json
import random

class ExampleLoader:
    def __init__(self, test_split: float=.4, file: str | Path = "./examples_simple.json", tokenize:bool = False, keys: list = None, expand: bool = False, expand_dict: dict = None):
        try:
            with open(file) as f:
                examples_dics = json.load(f)
                if tokenize and keys is not None:
                    self.tokenize_examples(examples = examples_dics, keys = keys, expand = expand, expand_dict= expand_dict)
                self.examples_all = [
                    Example(suffix = e['index'],
                            series_description = e['SeriesDescription'],
                            protocol_name = e['ProtocolName'],
                           # task_name = e['TaskName'],
                            reptition_time = e['RepetitionTime'],
                            echo_time = e['EchoTime'],
                            inversion_time = e['InversionTime'],
                            pulse_sequence_type = e['PulseSequenceType'],
                            flip_angle = e['FlipAngle'],
                            manufacturer = e['Manufacturer'],
                            model = e['ManufacturersModelName'],
                    )
                    for e 
                    in examples_dics
                    ]
                if not (0 < test_split < 1):
                    raise(ValueError("Split value most be between 0 and 1"))
                self.test_split = test_split               
        except FileNotFoundError:
            raise(FileNotFoundError("File Not Found"))
    
    def filter_types(self, types: list,):
        pass

    def tokenize_examples(self, examples: list, keys: list = None, expand: bool = False, expand_dict: dict = None):
        if keys == None:
            return
        for key in keys:
            for example in examples:
                tokens = self._tokenize(example[key])
                if expand and expand_dict is not None:
                    tokens = _expand_from_dict(tokens, expand_dict)
                example[key] = ' '.join(tokens)

    def _tokenize(self, string: str):
        tokens = []
        current_token = ''
        for char in string:
            if char.isupper():
                if current_token:
                    if current_token[-1].isupper() or current_token[-1].isdigit():
                        current_token +=char
                        continue
                    else:
                        tokens.append(current_token)
                current_token = char
            elif char == "_":
                if current_token:
                    tokens.append(current_token)
                current_token = ''
            elif char == "-":
                if current_token:
                    tokens.append(current_token)
                current_token = ''
            elif char == " ":
                if current_token:
                    tokens.append(current_token)
                current_token = ''
            else:
                current_token += char
        if current_token:
            tokens.append(current_token)
        return tokens

    def _expand_from_dict(tokens: list, expand_dict: dict):
        raise NotImplementedError("exapnd from dict is not implemented")

    def get_splits(self, randomize: bool = False, test_split: float | None = None):
        if randomize:
            random.shuffle(self.examples_all)
        if test_split is None:
            test_split = self.test_split
        ind = int(len(self.examples_all)*test_split)
        self.examples_test = self.examples_all[:ind]
        self.examples_store = self.examples_all[ind:]
        return self.examples_test, self.examples_store
