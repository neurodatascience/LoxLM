from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
from pydantic import Field
from pydantic import validator


class Example(BaseModel):
    """Example pydantic class."""

    suffix: str = Field(description="The BIDs suffix corresponding to the DICOM fields")
    series_description: str = Field(
        description="DICOM field for series description. Inputted by user."
    )
    protocol_name: str = Field(description="DICOM field for protocol name. Inputted by user.")
    task_name: str = Field(default="NA", description="DICOM field for task name. Inputted by user.")
    repetition_time: float = Field(
        default=float("nan"), description="DICOM field for repetition time. Inputted by user."
    )

    echo_time: float = Field(
        default=float("nan"), description="DICOM field for echo time. Inputted by user."
    )

    inversion_time: float = Field(
        default=float("nan"), description="DICOM field for inversion time. Inputted by user."
    )
    pulse_sequence_type: str = Field(
        default="NA", description="DICOM field for pulse sequence type. Inputted by user."
    )
    flip_angle: float = Field(
        default=float("nan"), description="DICOM field for flip angle. Inputted by user."
    )
    manufacturer: str = Field(
        default="NA", description="DICOM field for scanner manufacturer. Inputted by user."
    )
    model: str = Field(default="NA", description="DICOM field for scanner model. Inputted by user.")

    @validator(
        "repetition_time", "echo_time", "inversion_time", "flip_angle", pre=True, always=True
    )
    def parse_float_na(cls, v):
        if isinstance(v, float):
            return float(v)
        else:
            return float("nan")

    @validator(
        "suffix",
        "series_description",
        "protocol_name",
        "task_name",
        "pulse_sequence_type",
        "manufacturer",
        "model",
        pre=True,
        always=True,
    )
    def parse_string(cls, v):
        if isinstance(v, str):
            return str(v)
        else:
            return "NA"


class BaseExampleRanker(ABC):
    """Abstract Example Ranker class."""

    def __init__(self, examples: list):
        self.examples = examples
        # self.clean_examples(examples)

    @abstractmethod
    def add_example(self, example):
        pass

    @abstractmethod
    def eval_distance(self, value):
        pass

    def clean_examples(self, examples: list):
        return [
            example
            for example in examples
            if example is not None and example != "NA" and example is not float("nan")
        ]


class FloatExampleRanker(BaseExampleRanker):
    """
    Example Ranker class for float data.

    Stores an array of floats in order of examples.
    Used to compare distance between each of the stored examples and
    a value inputted to eval_distance.
    """

    def __init__(self, examples: list):
        # self.examples = np.log(np.array(examples))
        self.examples = np.array(examples)

    def add_example(self, example: float):
        if example is not None and example != "NA" and example is not float("nan"):
            # self.examples = np.append(self.examples,np.log(example))
            self.examples = np.append(self.examples, example)
        else:
            self.examples = np.append(self.examples, np.nan)

    def eval_distance(self, val: float):
        # distances = [abs(val-example) if not np.isnan(example) else np.nan for example in self.examples]
        distances = np.where(np.isnan(self.examples), np.abs(self.examples - val), np.nan)
        return self.normalize(distances)

    def normalize(self, values):
        min_val = np.min(values)
        max_val = np.max(values)
        if min_val == max_val:
            return np.zeros_like(values)
        else:
            return (values - min_val) / (max_val - min_val)


class SemanticExampleRanker(BaseExampleRanker):
    """
    Example Ranker for string data.

    Similar to float example ranker, this class stores the
    vector embeddings of a list of example strings. It then
    is used to calculate distance between each of these vectors
    and an inputted string.

    Model is either sentence transformer or OpenAIEmbedding
    """

    def __init__(self, examples: list, model):
        super().__init__(examples)

        self.openai = isinstance(model, OpenAIEmbeddings)

        self.model = model
        if not self.examples:
            self.embeddings = np.array()
        else:
            self.embeddings = self.embed(self.examples)

    def add_example(self, example):
        self.examples.append(example)
        if example is not None and example != "NA":
            if self.openai:
                self.embeddings + self.model.embed_query(example)
            else:
                self.embeddings + self.model.encode([example])[0]
        else:
            self.embeddings + np.nan

    def eval_distance(self, example):
        if self.openai:
            query_embedding = self.model.embed_query(example)
            distances = np.dot([query_embedding], np.array(self.embeddings).T)
            # print(distances[1])
            return self.normalize(distances[0])
        else:
            query_embedding = self.model.encode([example])[0]
            distances = self.model.similarity([query_embedding], self.embeddings)
            distances = distances[0]
            return self.normalize(distances.numpy())

    def embed(self, examples):
        if self.openai:
            return self.model.embed_documents(examples)
        else:
            return self.model.encode(examples)

    def normalize(self, values):
        min_val = np.min(values)
        max_val = np.max(values)
        if min_val == max_val:
            return np.zeros_like(values)
        else:
            return (values - min_val) / (max_val - min_val)


class MultiExampleSelector(BaseExampleSelector):
    """
    Combines semantic and float example rankers.

    It is used to select the 'k' nearest stored examples to an input query.
    """

    def __init__(
        self,
        model,
        examples: [Example],
        k: int = None,
    ):
        self.examples = examples
        self.k = k
        if examples is not None and len(examples) > 0:
            self.fields = examples[0].__fields__
            self.field_rankers = {}
            for field, d in self.fields.items():
                values = [example.dict()[field] for example in examples]
                if d.annotation is str:
                    self.field_rankers[field] = SemanticExampleRanker(values, model)
                elif d.annotation is float:
                    self.field_rankers[field] = FloatExampleRanker(values)
                else:
                    raise (TypeError(f"Unexpected type: {d.annotation}"))

        else:
            raise (Exception("give me more stuff"))

    def add_example(self, example: Example):
        self.examples.append(example)
        for field, field_ranker in self.field_rankers.items():
            if example.field is not None and example.field != "NA":
                field_ranker.add_example(example)

    def select_examples(self, input: Example, weights: dict | None = None, k: int = 3):
        """
        Return the k examples with smalled distance to input Example.

        Parameters
        ----------
        input: Example
            The example that is we are trying to find other examples similar too.

        weights: dict
            A dictionary with keys corresponding to keys in the Example class. The values should be floats.
            This is used to weight some fields more strongly than others.

        k: int
            The number of examples to return
        """
        if isinstance(input, dict):
            input = Example.construct(**input)
        dist = np.zeros(shape=(0, len(self.examples)))
        for field, value in input.model_dump().items():
            try:
                if np.isnan(value):
                    continue
            except Exception:
                pass
            if value is not None and value != "NA" and field != "suffix":
                field_ranker = self.field_rankers[field]
                distances = field_ranker.eval_distance(value)
                if weights is not None and weights[field] is not None:
                    distances = distances * weights[field]
                dist = np.vstack([dist, distances])
        # self.plot(dist)
        num_distances = np.sum(~np.isnan(dist), axis=0)
        final_distances = np.nansum(dist, axis=0)
        final_distances = final_distances / num_distances
        example_dist = zip(self.examples, final_distances.tolist())
        sorted_examples = sorted(example_dist, key=lambda x: x[1])
        test = sorted_examples[:k]
        exs = [example[0].model_dump() for example in test]
        return exs

    def plot(self, data):
        """Visualize distribution of distances."""
        import matplotlib.pyplot as plt

        num_rows, num_cols = data.shape
        fig, axs = plt.subplots(num_rows, 1, figsize=(8, 2 * num_rows))  # Adjust figsize as needed

        for i in range(num_rows):
            axs[i].scatter(np.arange(num_cols), data[i], marker="o", label=f"Row {i}")
            axs[i].set_xlabel("Index")
            axs[i].set_ylabel("Value")
            axs[i].set_title(f"Scatter Plot for Row {i}")
            axs[i].legend()
            axs[i].grid(True)
        plt.tight_layout()  # Optional: Adjust layout
        plt.show()
