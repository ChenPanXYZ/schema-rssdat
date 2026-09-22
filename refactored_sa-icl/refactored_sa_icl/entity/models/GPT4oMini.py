# Model is a simple class that handles the API requests to different end services.
import json
from openai import OpenAI
import os

from refactored_sa_icl.entity.models.Model import Model
from credentials import OPENAI_CREDENTIAL

import traceback

class GPT4oMini(Model):
    api_key: str
    def __init__(self, ctx_len: int=512):
        # load from environment variable.
        self.ctx_len = ctx_len
        self.api_key = OPENAI_CREDENTIAL
        # self.ctx_len = ctx_len

    def interact(self, messages: str, **kwargs) -> dict:
        # require a json_format, temperature, e.t.c
        try:
            json_format = kwargs.get("json_format")
            temperature = kwargs.get("temperature")
            top_p = kwargs.get("top_p", 1.0)
            max_tokens = kwargs.get("max_tokens", 4096)

            messages = [
                {"role": "user", "content": messages}
            ] if isinstance(messages, str) else messages

            if json_format is None:
                raise ValueError("json_format is required")
            if temperature is None:
                raise ValueError("temperature is required")

            # call the API and return the response.
            import openai
            client = OpenAI(api_key=self.api_key)
            prompt_input = {
                "model": "gpt-4o-mini-2024-07-18",
                "messages": messages,
                "response_format": json_format,
                "temperature": temperature,
                "top_p": top_p,
                "n": 1,
                "max_tokens": max_tokens,
                "logprobs": True,  # enable logprobs
                "top_logprobs": 5  # get top-N alternative token probabilities
            }

            # TODO: calculate the price.

            response = client.beta.chat.completions.parse(
                **prompt_input
            )

            # Inspect logprobs
            tokens = []
            for choice in response.choices:
                for block in choice.logprobs.content:
                    tokens.append((block.token, block.logprob))

            # Find final_answer tokens
            inside = False
            final_answer_tokens = []
            for token, logprob in tokens:
                if '"final_answer"' in token:
                    inside = True
                    continue
                if inside:
                    if token.strip() in [",", "}"]:  # end of the field
                        break
                    final_answer_tokens.append((token, logprob))

            # Convert logprobs to probabilities
            final_answer_probs = [(t, 10 ** lp) for t, lp in final_answer_tokens]

            for token, prob in final_answer_probs:
                print(f"{token!r}: {prob:.4f}")

            # return a json of the response.
            response = json.loads(response.choices[0].message.content)
            return response

        except Exception as e:
            # trace the error
            print(f"Error in GPT4oMini interact: {e}")
            traceback.print_exc()
