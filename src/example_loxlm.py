from __future__ import annotations

from loxlm import LoxLM

# To test using existing examples. Will output a text file of the prompts used.
# Will prompt for an open ai api key.
loxlm = LoxLM(test=True, log_prompts=True)

loxlm.batch_input()
