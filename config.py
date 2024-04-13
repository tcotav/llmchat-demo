import openai
import os

openai.api_key  = os.environ['OPENAI_API_KEY']
#llm_name = "gpt-4"
llm_name = "gpt-3.5-turbo"
persist_directory = 'docs/chroma/'