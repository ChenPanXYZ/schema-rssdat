from pydantic import BaseModel


class Schema(BaseModel):
    # TODO: define the attributes as strings for your schema here.


class Response(BaseModel):
    knowledge_schema: Schema

SCHEMA_PROMPT = # TODO: how you want to tell the LLM to generate a schema for a given information.


SCHEMA_SOLVER_PROMPT = SCHEMA_PROMPT

# Below is not needed any more, but keep them as None.
SCHEMA_SAMPLE_QUESTION = None
sample_schema = None
schema = None
summary = None
SCHEMA_SAMPLE_RESPONSE = None
