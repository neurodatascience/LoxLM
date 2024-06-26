from langchain_core.example_selectors.base import BaseExampleSelector
import statistics
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from typing import Optional, Union
import numpy as np
from pydantic import BaseModel, validator
import typing
import scipy
from sklearn import preprocessing

class Example(BaseModel):
        index: str
        series_description: str
        protocol_name: str
        task_name: str =  float('nan')
        repetition_time: float = float('nan')
        echo_time: float = float('nan')
        inversion_time: float = float('nan')
        pulse_sequence_type: str = None
        flip_angle: float = float('nan')
        manufacturer: str = None
        model: str = None
    #typing or whatever for an example class

        @validator('repetition_time','echo_time','inversion_time','flip_angle',
                pre=True, always=True)
        def parse_float_na(cls,v):
            if isinstance(v,float):
                return float(v)
            else:
                return float('nan')
class BaseExampleRanker(ABC):
    def __init__(self, examples: list):
        self.examples = examples
        #self.clean_examples(examples)

    @abstractmethod
    def add_example(self, example):
        pass

    @abstractmethod
    def eval_distance(self, value):
        pass

    def clean_examples(self, examples: list):
        return [example for example in examples if example is not None and
                example != "NA" and example is not float('nan')]


class FloatExampleRanker(BaseExampleRanker):
    def __init__(self,examples: list):
        super().__init__(examples)


    def add_example(self, example: float):
        if (example is not None and example != "NA" and example is not
        float('nan')):
            self.examples.append(example)
        else:
            self.examples.append(np.nan)


    def eval_distance(self, val: float):
        distances = [abs(val-example) if not np.isnan(example) else np.nan for example in self.examples]
        return self.normalize(distances)

    def normalize(self, values):
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            return np.zeros_like(values)
        else:
            return [(val-min_val)/(max_val-min_val) for val in values]



class SemanticExampleRanker(BaseExampleRanker):
    def __init__(self, examples: list, model):
        super().__init__(examples)
        self.model = model
        if not self.examples:
            self.embeddings = []
        else:
            self.embeddings = self.embed(self.examples)

    def add_example(self, example):
        self.examples.append(example)
        if example is not None and example != "NA":
             self.embeddings + self.model.encode([example])[0]
        else:
            self.embeddings +np.nan

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


    def select_examples(self, input: Example, k: int =3):
        dist = np.ndarray(shape=(1,len(self.examples)))
        for field, value in input.model_dump().items():
            if value is not None and value != "NA" and field !='index':
                field_ranker = self.field_rankers[field]
                distances = field_ranker.eval_distance(value)
                distances = np.array(distances).reshape(1,-1)
                dist = np.vstack([dist, distances])
        num_distances = dist.shape[0]
        final_distances = np.sum(dist, axis=0)
        final_distances = final_distances / num_distances
        example_dist = zip(self.examples, final_distances.tolist())
        sorted_examples = sorted(example_dist, key =lambda x: x[1])
        print(sorted_examples[:10])
        return sorted_examples[:k]



