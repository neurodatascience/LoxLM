from __future__ import annotations

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from src.utils.multi_example_selector import FloatExampleRanker
from src.utils.multi_example_selector import SemanticExampleRanker


@pytest.fixture
def model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


@pytest.fixture
def examples_full():
    return


@pytest.fixture
def examples_float_dirty():
    return [1.0, 2.0, "NA", None, 3.0]


@pytest.fixture
def examples_string_dirty():
    return


@pytest.fixture
def examples_string():
    return ["cat", "dog", "hat"]


@pytest.fixture
def examples_float():
    return [1.0, 2.0, 3.0]


@pytest.fixture
def float_ranker(examples_float):
    return FloatExampleRanker(examples_float)


@pytest.fixture
def semantic_ranker(examples_string, model):
    return SemanticExampleRanker(examples_string, model)


def test_float_add_example(examples_float):
    ranker = FloatExampleRanker(examples_float)
    ranker.add_example(4.0)
    assert ranker.examples == np.array([1.0, 2.0, 3.0, 4.0])


def test_normalize(examples_float, float_ranker):
    assert float_ranker.normalize(examples_float) == np.array([0.0, 0.5, 1.0])


def test_eval_distance(float_ranker):
    assert float_ranker.eval_distance(2.0) == np.array([1.0, 0.0, 1.0])


def test_semantic_add_example(semantic_ranker):
    semantic_ranker.add_example("couch")
