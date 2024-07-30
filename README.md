# LoxLM


## Installation
git clone the repository

in the directory 'pip install .'

## Usage

LoxLM can be used as an api. Look at example_loxlm.py for a basic testing usage script.

### Local

To run the models locally you need to install ollama. You can do this in a docker image or not.

Once you have ollama running you need to run the command 'ollama pull <model-name>'. Model name  by default is gemma2
but you can use whatever model you like by altering the python file.

### OpenAI

OpenAI usage is very simple. Simply enter you open ai api key when prompted.

### Context

If running with context you must spin up the milvus docker containers. With docker install run
'''docker compose up -d" in the main directory.
