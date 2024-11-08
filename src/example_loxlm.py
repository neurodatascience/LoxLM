from __future__ import annotations

from loxlm import LoxLM

# To test using existing examples. Will output a text file of the prompts used.
# Will prompt for an open ai api key.
loxlm = LoxLM(test=True, log_prompts=True)

loxlm.batch_input()
"""
#If you want to perform inference on you own dataset.
input_dicoms = "replace with path to your DICOMs"
output_file = "replace with your desired output file"

loxlm = LoxLM(test=False,input_dicoms = input_dicoms, output_file=output_file openai=True)
loxlm.batch_input()
#or you could do
loxlm = LoxLM(test=False, openai=True)
loxlm.batch_input(input_dicoms=input_dicoms, output_file=output_file)
"""
