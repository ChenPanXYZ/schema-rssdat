# import json
# import anthropic
# from src.entity.models.Model import Model
#
# from credentials import CLAUDE_CREDENTIAL
#
#
# # Function to convert OpenAI's json_schema to Claude's input_schema
# def convert_to_claude_schema(openai_schema):
#     # Extract properties
#     result = {
#         "name": "response_format",
#         "description": "The format of the response",
#         "input_schema":
#             {
#                 "type": "object",
#                 "properties": openai_schema['json_schema']['schema']["properties"],
#                 "required": openai_schema['json_schema']['schema']["required"],
#                 "additionalProperties": False,
#             }
#     }
#     return result
#
#
#     # Create Claude's input_schema
#     claude_input_schema = {}
#     for key, value in properties.items():
#         # Map OpenAI types to Claude types
#         if value["type"] == "string":
#             claude_input_schema[key] = "string"
#         elif value["type"] == "integer":
#             claude_input_schema[key] = "integer"
#         # Add more type mappings as needed
#
#     return claude_input_schema
#
# class Claude(Model):
#     '''
#     The Claude model class.
#     '''
#
#     api_key: str
#     ctx_len: int
#     def __init__(self, ctx_len: int=512):
#         # load from environment variable.
#         self.api_key = CLAUDE_CREDENTIAL
#         self.ctx_len = ctx_len
#
#     def interact(self, messages: str, **kwargs):
#         '''
#         When the message is a string, append json schema to the string
#         When message is a list of dictionaries, append json schema to the user message
#         '''
#         # require a json_format, temperature, e.t.c
#         # This is a json schema
#         json_format = kwargs.get("json_format")
#         temperature = kwargs.get("temperature")
#
#         messages = [
#             {"role": "user", "content": messages},
#         ] if isinstance(messages, str) else messages
#
#         if json_format is None:
#             raise ValueError("json_format is required")
#         if temperature is None:
#             raise ValueError("temperature is required")
#
#         # remove message whose's role is system from messages, and assign the value to a variable.
#         system_message = [message['content'] for message in messages if message["role"] == "system"][0]
#         messages = [message for message in messages if message["role"] != "system"]
#         # call the API and return the response.
#         client = anthropic.Anthropic(api_key=self.api_key)
#         prompt_input = {
#             "model": "claude-3-5-sonnet-20241022",
#             "temperature": temperature,
#             "top_k": 1,
#             "tools": [convert_to_claude_schema(json_format) ],
#             "tool_choice": {"type": "tool", "name": "response_format"},
#             "system": system_message,
#             "messages": messages,
#             "max_tokens": self.ctx_len
#         }
#         # TODO: calculate the price.
#         response = client.messages.create(
#             **prompt_input
#         )
#         response = json.loads(response.json())['content'][0]['input']
#
#         return response