import json
from pathlib import Path
from langchain_core.prompt_values import PromptValue

class PromptLogger:
    """Class to save prompts and write them to a text file.
    
        file: str
            Address of text file to be written.
    """
    def __init__(self, file: str = "./prompt_logs.txt"):
        self.file = file
        self.prompts = []
        self.inputs = []

    def add_prompt(self, prompt: PromptValue, input: dict | None = None):
        messages = prompt.to_messages()
        pretty_messages = [p.pretty_repr() for p in messages]
        self.prompts.append(pretty_messages)

        self.inputs.append(input)

    def write_logs(self):

        with open(self.file, "w") as f:
            for prompt, input in zip(self.prompts,self.inputs):
                string = [f'Input: {input}']
                string.append("Prompt:")
                for line in prompt:
                    string.append(line)
                for s in string:
                    f.write(s)
                    f.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

