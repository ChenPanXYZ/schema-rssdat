import json
from openai import OpenAI
# Assuming these imports exist in your project structure
from refactored_sa_icl.entity.models.Model import Model
from credentials import OPENAI_CREDENTIAL
from pydantic import BaseModel


class Qwen3(Model):
    api_key: str

    def __init__(self, ctx_len: int = 4096):
        self.api_key = 'EMPTY'
        self.ctx_len = ctx_len

    def interact(self, messages: str | list, **kwargs):
        # Required parameters
        json_format = kwargs.get("json_format")
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p", 1.0)
        max_tokens = kwargs.get("max_tokens", self.ctx_len)

        if temperature == 0:
            temperature = 0.6

        if json_format is None:
            raise ValueError("json_format is required")
        if temperature is None:
            raise ValueError("temperature is required")

        # 1. Normalize single string input to list format
        messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )

        # =================================================================
        # FIX: Flatten "content" lists into strings
        # The server crashes if "content" is a list (e.g. [{"type": "text", ...}])
        # =================================================================
        sanitized_messages = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                # Join all text parts into a single string
                full_text = "".join(
                    part["text"] for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
                sanitized_messages.append({**msg, "content": full_text})
            else:
                sanitized_messages.append(msg)

        messages = sanitized_messages
        # =================================================================

        # Local Qwen server
        client = OpenAI(
            api_key=self.api_key,
            base_url="http://0.0.0.0:34568/v1"
        )

        # Convert BaseModel to JSON schema if needed
        # Note: client.beta.chat.completions.parse usually takes the class directly,
        # but if your backend requires a schema dict, this is fine.
        if isinstance(json_format, type) and issubclass(json_format, BaseModel):
            json_format = {
                "type": "json_object",
                "schema": json_format.model_json_schema()
            }

        # Build the prompt
        prompt_input = {
            "model": "qwen3-8b",
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
            "max_tokens": max_tokens,
            "response_format": json_format,
        }

        # Call Qwen API
        response = client.beta.chat.completions.parse(**prompt_input)

        # Parse JSON response
        raw = response.choices[0].message.content
        return json.loads(raw.replace('\n', ' '))
