# import Model class.
from src.entity.models.Model import Model
from src.entity.problems.Problem import Problem
from pydantic import BaseModel


class Schema(BaseModel):
    insights: str


class SummarySchemaGenerator:
    def __init__(self, model: Model, knowledge_base):
        self.model = model
        self.prompt_template = """
# Task:
Your task is to come up with meaningful insights from the following question and its answer, in a bullet-point
format.
{previous_problems_answers}
"""
        self.response_format = self.create_response_format()
    def generate_schema(self, past_knowledge_string: str) -> str:

        prompt = self.prompt_template.format(past_knowledge=past_knowledge_string)

        response = self.model.interact(prompt, json_format=self.response_format, temperature=0)
        parsed_response = Schema(**response)
        return parsed_response.insights

    def create_response_format(self):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "output_schema",
                "schema": Schema.model_json_schema()
            }
        }