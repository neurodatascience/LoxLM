import json
from pathlib import Path

from loxlm.utils.multi_example_selector import Example


class ExampleLoader:
    def __init__(self, test_split: float=.8, file: str | Path = "/home/rand/github/LoxLM/LoxLM/src/utils/example_loader.py",):
        try:
            with open(file) as f:
                examples_all = json.load(f)
                examples_all = [
                    Example(index = e['index'],
                            series_description = e['SeriesDescription'],
                            protocol_name = e['ProtocolName'],
                            task_name = e['TaskName'],
                            reptition_time = e['RepetitionTime'],
                            echo_time = e['EchoTime'],
                            inversion_time = e['InversionTime'],
                            pulse_sequence_type = e['PulseSequenceType'],
                            flip_angle = e['FlipAngle'],
                            manufacturer = e['Manufacturer'],
                            model = e['ManufacturersModelName'],
                            )
                    for e
                    in examples_all
                    ]
                if not (0 < test_split < 1):
                    raise(ValueError("Split value most be between 0 and 1"))
                ind = int(len(examples_all)*test_split)
                self.examples_test = examples_all[:ind]
                self.examples_store = examples_all[ind:]

        except FileNotFoundError:
            raise(FileNotFoundError("File Not Found"))

    def filter_types(self, types: list,):
        pass

    def get_splits(self,):
        return self.examples_test, self.examples_store
