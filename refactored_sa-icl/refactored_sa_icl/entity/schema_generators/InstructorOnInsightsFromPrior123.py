# import Model class.
from abc import abstractmethod
import json

from components.instructors.Instructor import Instructor
from components.instructors.retriever.abstractRetriever import AbstractRetriever
from components.models.Model import Model
from components.problems.Problem import Problem


class InstructorOnInsightsFromPrior(Instructor):
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
        prompt = self.prompt_template.format(target_problem=str(problem),
                                             previous_problems_answers=previous_problems_answers)
        instruction = self.model.interact(prompt,
                                          json_format=self.response_format, temperature=0)

        _ = json.loads(instruction.choices[0].message.content)
        return past_knowledges, _["refined_insights"], similarity, _["insights"]

    # find the most relevant problems to the target problem
    def run_retrieval(self, problem, **kwargs) -> Problem:
        try:
            prior_experience_index_matrix, similarity_matrix = self.retreiver.run_retrieval([problem])
            from exp_config import UTILIZED_KNOWLEDGE_INDEX_START, UTILIZED_KNOWLEDGE_INDEX_END, EXCLUDE_SELF
            prior_experience_index = prior_experience_index_matrix[0][UTILIZED_KNOWLEDGE_INDEX_START:
                                                                      UTILIZED_KNOWLEDGE_INDEX_END] # TODO: get
            # second index.
            similarity = similarity_matrix[0][prior_experience_index]
            # similar_indices = similar_indices[1:]
            # print(Dataset.get_all_problems()[0])
            relevant_problem = [self.all_problems[i] for i in prior_experience_index]
            my_index = Problem.get_problem_index(problem)
            print(f'{my_index} v.s. {prior_experience_index} about {similarity}')
            if my_index in prior_experience_index and EXCLUDE_SELF:
                raise ValueError("The retrieved problem is the same as the target problem.")

            if len(relevant_problem) == 0:
                return None, None
            # try to use LLM to rerank the problems.
            prompt="""You are a chemist. Consider the following target question:
{problem}
### Task:
You have been given the following 5 questions. They are ranked randomly now. Your task is to find the question that is 
most relevant to the target question, and report the label of that question.
{relevant_problems}"""
            json_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "relevant_problem",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "relevant_problem_label":{
                                "type": "string",
                                "description": "The label of the most relevant problem to the target problem.",
                                "enum": [str(i + 1) for i, problem in enumerate(relevant_problem)]
                            }
                        },
                        "required": ["relevant_problem_label"],
                        "additionalProperties": False,
                    },
                },
            }

            relevant_problem_string = ""
            for i, problem in enumerate(relevant_problem):
                relevant_problem_string += """--- Relevant Problem Label {i} ---
{problem}
--- End of Relevant Problem Label {i} ---
                """.format(i=i+1, problem=str(problem))
                if i != len(relevant_problem) - 1:
                    relevant_problem_string += "\n"

            prompt = prompt.format(problem=str(problem), relevant_problems=relevant_problem_string,
                                                            )
            response = self.model.interact(prompt,
                                                json_format=json_format, temperature=0)
            response = json.loads(response.choices[0].message.content)
            relevant_problem = [relevant_problem[int(response["relevant_problem_label"]) - 1]]
            similarity = [similarity[int(response["relevant_problem_label"]) - 1]]
            print("LLM's done reranking.")
            print(int(response["relevant_problem_label"]) - 1)
            return relevant_problem, similarity
        except Exception as e:
            print(e)
            return None, None