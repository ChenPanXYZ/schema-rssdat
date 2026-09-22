# Model is a simple class that handles the API requests to different end services.
import json

# import Model class
from openai import OpenAI
from refactored_sa_icl.entity.models.Model import Model
from credentials import OPENAI_CREDENTIAL
from pydantic import BaseModel

class Llama3(Model):
    api_key: str

    def __init__(self, ctx_len: int = 4096):
        self.api_key = 'EMPTY'
        self.ctx_len = ctx_len

    def interact(self, messages: str, **kwargs):
        # required params
        json_format = kwargs.get("json_format")
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p", 1.0)
        max_tokens = kwargs.get("max_tokens", self.ctx_len)

        if temperature == 0:
            temperature = 0.6 # Note that we decided temperature shouldn't be 0 for llama3.

        if json_format is None:
            raise ValueError("json_format is required")
        if temperature is None:
            raise ValueError("temperature is required")

        # Normalize messages
        messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )

        # Local Llama server client
        client = OpenAI(
            api_key=self.api_key,
            base_url="http://0.0.0.0:34567/v1"
        )

        # Convert BaseModel to OpenAI JSON schema
        if issubclass(json_format, BaseModel):
            json_format = {
                "type": "json_object",
                "schema": json_format.model_json_schema()
            }

        prompt_input = {
            "model": "llama3.1-8b",
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
            "max_tokens": max_tokens,
            "response_format": json_format,
        }

        response = client.beta.chat.completions.parse(**prompt_input)

        # Parse returned JSON text
        raw_text = response.choices[0].message.content
        return json.loads(raw_text.replace('\n', ' '))
