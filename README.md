# LoxLM


## Installation
git clone the repository
```git clone https://github.com/neurodatascience/LoxLM.git```
enter the LoxLM directory and then install. It is recommended to enter a virtual environment before installing.
i.e. ```python -m venv loxlm```
    ```source loxlm/bin/activate```
    ```pip install . ```

## Usage

LoxLM can be used as an api. Look at example_loxlm.py for a basic testing usage script.
Import the package as '''from loxlm import LoxLM'''
You can create a LoxLM object and then invoke input using the input_batch method.
You can either run LoxLM with a local LLM using ollama or using an OpenAI API key.
It is recommended to use OpenAI as of August 2024.
### Local LLM

To run the models locally you need to [install ollama](https://ollama.com/download). You can do this in a docker image or not.
Here are instructions to run ollama as a docker image. [https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image](https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image)
Once you have ollama running you need to run the command 'ollama pull <model-name>'. Model name  by default is gemma2
but you can use whatever model you like by altering the python file.

### OpenAI LLM (recommended)

OpenAI usage is very simple. Simply enter you open ai api key when prompted. You can also set an environment variable to avoid this step.
Be careful not accidentally upload an API key to the internet.

### Context (not recommended)

If running with context you must spin up the milvus docker containers. With docker install run
'''docker compose up -d" in the main directory.

### LoxLM object
The LoxLM has several parameters. By default openai will be used and no context retrieval databases will be installed. By default it is in test mode which means that the example database will be split so that some are used for the example database and other are used as testing inputs.
