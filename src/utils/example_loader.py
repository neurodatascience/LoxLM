
from pathlib import Path
import json
class ExampleLoader:
    def __init__(self, test_split: float=.5, file: str | Path = "../examples_all.json",):
        try:
            with open(file) as f:
                examples_all = json.load(f)
                examples_all = [
                    {
                        "h": f"SeriesDescription: {e['SeriesDescription']}\nProtocolName: {e['ProtocolName']}",
                        "bot": f"Suffix: {e['index']}"
                        } 
                    for e in examples_all
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
