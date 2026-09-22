# Model is a simple class that handles the API requests to different end services.
import json

# import Model class
from openai import OpenAI
from refactored_sa_icl.entity.models.Model import Model
from credentials import OPENAI_CREDENTIAL

class GPT4o(Model):
    api_key: str

    def __init__(self, ctx_len: int = 512):
        self.api_key = OPENAI_CREDENTIAL

    def interact(self, messages: str, **kwargs):
        # Extract required parameters
        json_format = kwargs.get("json_format")
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p", 1.0)
        max_tokens = kwargs.get("max_tokens", 4096)

        if json_format is None:
            raise ValueError("json_format is required")
        if temperature is None:
            raise ValueError("temperature is required")

        # Normalize message input type
        messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )

        # OpenAI API call
        client = OpenAI(api_key=self.api_key)

        prompt_input = {
            "model": "gpt-4o-2024-11-20",
            "messages": messages,
            "response_format": json_format,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
            "max_tokens": max_tokens,   # TODO: I set the max token for gpt 4o as well.
        }

        response = client.beta.chat.completions.parse(**prompt_input)

        # Return parsed JSON
        return json.loads(response.choices[0].message.content)
