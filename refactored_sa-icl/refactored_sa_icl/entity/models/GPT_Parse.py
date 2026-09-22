# Model is a simple class that handles the API requests to different end services.
import json
import os
from dotenv import load_dotenv
from credentials import *

# import Model class
from openai import OpenAI
from refactored_sa_icl.entity.models.Model import Model

load_dotenv()

class GPT_Parse(Model):
    api_key: str
    def __init__(self, ctx_len: int=512):
        # load from environment variable.
        self.api_key = OPENAI_CREDENTIAL
        # self.ctx_len = ctx_len

    def interact(self, model, messages: str, **kwargs):
        # require a json_format, temperature, e.t.c

        text_format = kwargs.get("text_format")
        temperature = kwargs.get("temperature")

        messages = [
            {"role": "user", "content": messages}
        ] if isinstance(messages, str) else messages

        if text_format is None:
            raise ValueError("text_format is required")
        if temperature is None:
            raise ValueError("temperature is required")

        # call the API and return the response.
        client = OpenAI(api_key=self.api_key)
        prompt_input = {
            "model": "gpt-4o-mini-2024-07-18",
            "input": messages,
            "text_format": text_format,
            "temperature": 0
        }
        # TODO: calculate the price.

        response = client.responses.parse(
            **prompt_input
        )
        # return a json of the response.
        response = response.output_parsed.model_dump()
        return response