# import Model class.
from abc import abstractmethod

from src.models.GPT4o import GPT4o
from src.models.GPT4oMini import GPT4oMini
from src.models.Model import Model
from src.problems.Problem import Problem
from pydantic import BaseModel


class Schema(BaseModel):
    insights: str


class SchemaGenerator:
    def __init__(self, model: Model, knowledge_base, retriever):
        self.model = model
        self.prompt_template = None
        self.response_format = self.create_response_format()
        self.retriever = retriever
        self.knowledge_base = knowledge_base


    def generate_schema(self, problem, past_knowledge) -> str:
        past_knowledge_string = self.format_past_knowledge(past_knowledge)
        problem_string = str(problem)

        prompt = self.prompt_template.format(target_problem=problem_string,
                                             past_knowledge=past_knowledge_string)
        
        response = self.model.interact(prompt, json_format=self.response_format, temperature=0)
        parsed_response = Schema(**response)
        return parsed_response.insights
    

    def run_retrieval(self, problem, **kwargs) -> Problem:
        try:
            past_knowledge_index_list, similarity_list = self.retreiver.sort_past_knowledge(problem)

            # TODO: what does UTILIZED_KNOWLEDGE_INDEX_START, UTILIZED_KNOWLEDGE_INDEX_END means
            from exp_config import UTILIZED_KNOWLEDGE_INDEX_START, UTILIZED_KNOWLEDGE_INDEX_END, EXCLUDE_SELF
            past_knowledge_index = past_knowledge_index_list[UTILIZED_KNOWLEDGE_INDEX_START: UTILIZED_KNOWLEDGE_INDEX_END]

            # second index.
            similarity = similarity_list[past_knowledge_index]
            relevant_problem = [self.all_problems[i] for i in past_knowledge_index]

            my_index = Problem.get_problem_index(problem)
            if my_index in past_knowledge_index and EXCLUDE_SELF:
                raise ValueError("The retrieved problem is the same as the target problem.")
            return relevant_problem, similarity
        except Exception as e:
            print(e)
            return None, None


    @abstractmethod
    def format_past_knowledge(self, past_knowledge):
        raise NotImplementedError
    

    def create_response_format(self):
        if type(self.model) == GPT4oMini or type(self.model) == GPT4o:
            return {
                "type": "json_schema",
                "json_schema":{
                "name": "output_schema",
                "schema": Schema.model_json_schema()
                }
            }
        else:
            raise ValueError("The model is not supported for this task.")