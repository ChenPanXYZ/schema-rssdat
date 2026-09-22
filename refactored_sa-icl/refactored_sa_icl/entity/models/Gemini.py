from google import genai
from google.genai import types
import json
import base64

# Assuming credentials and Model are in your path
from credentials import GEMINI_CREDENTIAL
from refactored_sa_icl.entity.models.Model import Model


class Gemini(Model):
    api_key: str
    ctx_len: int  # Renamed from ctx_len for clarity

    def __init__(self, ctx_len: int = 512):
        # Default increased to 4096 to prevent JSON cut-off (Unterminated string error)
        self.api_key = GEMINI_CREDENTIAL
        self.ctx_len = ctx_len

        # Initialize Client once
        self.client = genai.Client(api_key=self.api_key)

    def _to_genai_content(self, input_data, role="user") -> list[types.Content]:
        """
        Universal adapter to convert diverse input formats (str, dict, list)
        into a strictly typed list[types.Content] for the SDK.
        """
        # 1. If it's already a Content object, wrap and return
        if isinstance(input_data, types.Content):
            return [input_data]

        # 2. If it's a list, process items into Parts
        if isinstance(input_data, list):
            parts = []
            for item in input_data:
                part = self._convert_item_to_part(item)
                if part:
                    parts.append(part)
            if not parts:
                return []
            return [types.Content(role=role, parts=parts)]

        # 3. Single item (dict or str)
        part = self._convert_item_to_part(input_data)
        if part:
            return [types.Content(role=role, parts=[part])]

        return []

    def _convert_item_to_part(self, item) -> types.Part | None:
        """Helper to convert atomic items into types.Part"""
        if isinstance(item, types.Part):
            return item

        if isinstance(item, str):
            return types.Part(text=item)

        if isinstance(item, dict):
            # Case: Standard Text
            if "text" in item:
                # Recursively handle if 'text' itself is a list (nested structure issue)
                content = item["text"]
                if isinstance(content, list):
                    # Flatten list of dicts/strings into one string
                    text_str = ""
                    for sub in content:
                        if isinstance(sub, dict):
                            text_str += sub.get("text", "")
                        elif isinstance(sub, str):
                            text_str += sub
                    return types.Part(text=text_str)
                return types.Part(text=str(content))

            # Case: Inline Media (Binary)
            if "data" in item and "mime_type" in item:
                return types.Part(
                    inline_data=types.Blob(
                        data=item["data"],
                        mime_type=item["mime_type"]
                    )
                )
        return None

    def interact(self, messages: list, **kwargs):
        """
        Main interaction method.
        messages: list of dicts [{'role': 'user', 'content': ...}]
        """
        # 1. Extract Parameters
        json_format = kwargs.get("json_format")
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p", 1.0)
        max_tokens = kwargs.get("ctx_len", self.ctx_len)

        if json_format is None:
            raise ValueError("json_format is required")
        if temperature is None:
            raise ValueError("temperature is required")

        # 2. Extract System Instruction (if any)
        # We look for role='system' and extract content
        system_message = next(
            (m["content"] for m in messages if m.get("role") == "system"),
            None
        )

        # 3. Build & Normalize Contents
        # Filter out system messages from the main conversation history
        normalized_contents = []
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                continue

            # Map 'assistant' to 'model' for Gemini SDK compatibility
            gemini_role = "model" if role == "assistant" else "user"

            # Use the adapter to safely convert content
            converted_content = self._to_genai_content(
                m.get("content"),
                role=gemini_role
            )
            normalized_contents.extend(converted_content)

        # 4. Configure Generation
        config = types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=json_format,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            system_instruction=system_message
        )

        # 5. Call API
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=normalized_contents,
                config=config
            )

            # 6. Robust Parsing
            # Gemini 2.0 with response_mime_type usually returns valid JSON text directly
            raw_text = response.text

            # Cleanup: Remove Markdown code fences if present (```json ... ```)
            if raw_text.strip().startswith("```"):
                lines = raw_text.strip().splitlines()
                # Remove first line (```json) and last line (```)
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines)

            return json.loads(raw_text)

        except json.JSONDecodeError:
            print("JSON Parsing Failed. Raw output:")
            try:
                print(response.text)
            except:
                print("No text in response.")
            return None

        except Exception as e:
            print(f"Gemini Interaction Error: {e}")
            return None
