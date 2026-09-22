# import Model class.
from abc import abstractmethod
import json

from overrides import overrides

from src.schema_generators.SchemaGenerator import SchemaGenerator
from src.retriever.abstractRetriever import AbstractRetriever
from src.models.GPT4o import GPT4o
from src.models.GPT4oMini import GPT4oMini
from src.models.Model import Model
from src.problems.Problem import Problem


class InstructorOnInsightsFromPriorNoRefinement(SchemaGenerator):
    def __init__(self, model: Model, retreiver: AbstractRetriever, all_problems):
        self.model = model
        from exp_config import INSTRUCTOR_PROMPT_TEMPLATE
        self.prompt_template = INSTRUCTOR_PROMPT_TEMPLATE
        self.retreiver = retreiver
        self.response_format = self.create_response_format()
        self.all_problems = all_problems

    # return an instruction (str)
    def write_instruction(self, problem, **kwargs) -> str:
        past_knowledge, similarity = self.run_retrieval(problem, **kwargs)
        if past_knowledge is None:
            return None, None, None, None

        previous_problems_answers = []
        past_knowledges = [past_knowledge] if type(past_knowledge) == Problem else past_knowledge
        for past_knowledge in past_knowledges:
            previous_problems_answers.append("""{past_knowledge}
Answer to this question is: {previous_problem_answer}""".format(past_knowledge=str(past_knowledge),
                                                                previous_problem_answer=past_knowledge.get_ground_truth()))
        previous_problems_answers = "\n".join(previous_problems_answers)

        problem_string = str(problem)
        prompt = self.prompt_template.format(target_problem=problem_string,
                                             previous_problems_answers=previous_problems_answers)
        instruction = self.model.interact(prompt,
                                          json_format=self.response_format, temperature=0)
        return _["insights"]

    # find the most relevant problems to the target problem

    @overrides
    def create_response_format(self):
        if type(self.model) == GPT4oMini or type(self.model) == GPT4o:
                return {
        "type": "json_schema",
        "json_schema": {
            "name": "instruction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "insights": {"type": "string", "description": "The insights from the previous problem."}
                },
                "required": ["insights"],
                "additionalProperties": False,
            },
        },
        }
        else:
            raise ValueError("The model is not supported for this task.")