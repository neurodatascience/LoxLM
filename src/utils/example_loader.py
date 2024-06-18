from pathlib import Path
import json
class ExampleLoader():
    def __init__(self, test_split: float=.5, file: str | Path = "../examples_all.json",):
        try:
            with open(file) as f:
                examples_all = json.load(f)
                examples_all = [
                    {
                        "h": f"SeriesDescription: {e['series_description']}\nProtocolName: {e['protocol_name']}",
                        "bot": f"Suffix: {e['suffix']}"
                        } 
                    for e in examples_all
                    ]
                if not (0 < test_split < 1):
                    raise(ValueError("Split value most be between 0 and 1"))
                examples_test = examples_all[:len(examples_all)*test_split]
                examples_store = examples_all[len(examples_all)*test_split:]
                return examples_test, examples_store
                
        except FileNotFoundError:
            print("File Not Found")
    
    def filter_types(self, types: list,):
        pass
