# Model is a simple class that handles the API requests to different end services.
import json
from mistralai import Mistral
import os

from refactored_sa_icl.entity.models.Model import Model
from credentials import MISTRAL_CREDENTIAL


class MistralSmall(Model):
    api_key: str

    def __init__(self, ctx_len: int = 512):
        self.ctx_len = ctx_len
        self.api_key = MISTRAL_CREDENTIAL

    def interact(self, messages: str, **kwargs) -> dict:
        # Required decoding parameters
        json_format = kwargs.get("json_format")
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p", 1.0)
        max_tokens = kwargs.get("max_tokens", 4096)

        if json_format is None:
            raise ValueError("json_format is required")
        if temperature is None:
            raise ValueError("temperature is required")

        # Normalize message type
        messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )

        # Mistral client
        client = Mistral(api_key=self.api_key)

        prompt_input = {
            "model": "mistral-small-2506",
            "messages": messages,
            "response_format": json_format,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
            "max_tokens": max_tokens,
        }

        response = client.chat.parse(**prompt_input)

        # Convert JSON response string → Python dict
        raw = response.choices[0].message.content
        return json.loads(raw)
