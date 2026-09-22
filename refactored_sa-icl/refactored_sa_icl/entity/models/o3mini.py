import json
import traceback
from openai import OpenAI
from refactored_sa_icl.entity.models.Model import Model
from credentials import OPENAI_CREDENTIAL


class o3mini(Model):
    def __init__(self, ctx_len: int = 128000):
        self.ctx_len = ctx_len
        self.api_key = OPENAI_CREDENTIAL

    def interact(self, messages, **kwargs) -> dict:
        try:
            # 1. EXTRACT PARAMS
            # Map standard 'max_tokens' to o3-mini's 'max_completion_tokens'
            # Default to 4096 to match your request
            max_tokens = kwargs.get("max_tokens", 4096)

            # Reasoning Effort: low, medium, high (Default: low to save cost/time)
            reasoning_effort = kwargs.get("reasoning_effort", "low")

            # JSON Schema is required for your framework
            json_format = kwargs.get("json_format")
            if json_format is None:
                raise ValueError("json_format is required")

            # 2. PREPARE CLIENT
            client = OpenAI(api_key=self.api_key)

            # Standardize messages (handle string input vs list input)
            messages = [
                {"role": "user", "content": messages}
            ] if isinstance(messages, str) else messages

            # 3. CONSTRUCT API PAYLOAD
            # CRITICAL: o3-mini uses 'max_completion_tokens'
            # STRICTLY REMOVED: temperature, top_p, logprobs (unsupported by o3-mini)
            prompt_input = {
                "model": "o3-mini",
                "messages": messages,
                "response_format": json_format,
                "reasoning_effort": reasoning_effort,
                "max_completion_tokens": max_tokens
            }

            # 4. EXECUTE REQUEST
            response = client.beta.chat.completions.parse(**prompt_input)

            # NOTE: Logprobs logic from GPT4oMini is skipped here because
            # o3-mini does not currently return logprobs.

            # 5. PARSE & RETURN RESPONSE
            # We match the exact return format of your GPT4oMini class:
            # strictly loading the content string as JSON.
            if response.choices[0].message.content:
                return json.loads(response.choices[0].message.content)

            # Fallback if content is empty but parsed object exists (edge case)
            elif response.choices[0].message.parsed:
                return response.choices[0].message.parsed.model_dump()

            return {}

        except Exception as e:
            print(f"Error in o3mini interact: {e}")
            traceback.print_exc()
            return {}
