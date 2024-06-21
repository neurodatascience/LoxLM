from langchain_core.example_selectors.base import BaseExampleSelector
import statistics
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from typing import Optional, Union
import numpy as np
from pydantic import BaseModel
import typing

class Example(BaseModel):
        series_description: str
        protocol_name: str
        task_name: str = None
        repitition_time: float = None
        echo_time: float = None
        inversion_time: float = None
        pulse_sequence_type: str = None
        flip_angle: float = None

    #typing or whatever for an example class

class BaseExampleRanker(ABC):
    def __init__(self, examples: list):
        self.examples = self.clean_examples(examples)

    @abstractmethod
    def add_example(self, example):
        pass

    @abstractmethod
    def eval_distance(self, value):
        pass

    def clean_examples(self, examples: list):
        return [example for example in examples if example is not None and examples != "NA"]
        

class FloatExampleRanker(BaseExampleRanker):
    def __init__(self,examples: list):
        super().__init__(examples)


    def add_example(self, example: float):
        if example is not None and example != "NA":
            self.examples.append(example)


    def eval_distance(self, val: float):
        distances = [abs(val-example) for example in self.examples]
        return self.normalize(distances)

    def normalize(self, values):
        min_val = min(values)
        max_val = max(values)
        return [(x - min_val)/(max_val - min_val) for x in values]




class SemanticExampleRanker(BaseExampleRanker):
    def __init__(self, examples: list, model):
        super().__init__(examples)
        self.model = model
        if not self.examples:
            self.embeddings = []
        else:
            self.embeddings = self.embed(self.examples)

    def add_example(self, example):
        if example is not None and example != "NA":
            self.examples.append(example)
            self.embeddings.append(self.model.encode([example])[0])

    def eval_distance(self, example):
        query_embedding = self.model.encode([example])[0]
        distances = self.model.similarity([query_embedding],self.embeddings)[0]
        return distances.tolist()

    def embed(self, examples):
        return self.model.encode(examples)
    

class MultiExampleSelector(BaseExampleSelector):
    def __init__(self, examples: [Example], k: int = None, model = SentenceTransformer("BAAI/bge-large-en-v1.5")):
        self.examples = examples
        self.k = k
        if examples is not None and len(examples)>0:
            self.fields = examples[0].__fields__
            self.field_rankers = {}
            for field, d in self.fields.items():
                values = [example.dict()[field] for example in examples]
                if  d.annotation is str:
                    self.field_rankers[field] = SemanticExampleRanker(values, model)
                elif d.annotation is float:
                    self.field_rankers[field] = FloatExampleRanker(values)
                else:
                    raise(TypeError(f"Unexpected type: {d.annotation}"))

        else:
            raise(Exception("give me more stuff"))


    def add_example(self, example: Example):
        self.examples.append(example)
        for field, field_ranker in self.field_rankers.items():
            if example.field is not None and example.field != "NA":
                field_ranker.add_example(example)

        #TODO: for all values of example that are floats update the mean and variance of examples

    def select_examples(self, input: Example, k: int =1):
        dist = np.ndarray()
        for field, value in input.dict().items():
            if value is not None and value != "NA":
                field_ranker = self.field_rankers[field]
                distances = field_ranker.eval_distance(value)
                distances = np.array(distances).reshape(1,-1)
                dist = np.vstack([dist, distances])
        num_distances = dist.shape[0]
        final_distances = np.sum(dist, axis=0)
        final_distances = final_distances / num_distances
        example_dist = zip(self.examples, final_distances.tolist())
        sorted_examples = sorted(example_dist, key =lambda x: x[1])
        return sorted_examples[:k]
                

        