# import Model class.
from abc import abstractmethod
import json

from src.schema_generators.SchemaGenerator import SchemaGenerator
from src.retriever.abstractRetriever import AbstractRetriever
from src.models.Model import Model
from src.problems.Problem import Problem


class InstructorOnInsightsFromPrior(SchemaGenerator):
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
#         problem_string = """
# Question: Who is credited with uniting the regions of Northern and Southern Erithia in 1335?
# Candidates:
#
# 	1.	King Eldric of Nordheim
# 	2.	Queen Lenora of Veridia
# 	3.	Emperor Hadrian of Solis
# 	4.	Lord Beltran of Westgate
# The answer to this question: 1.	King Eldric of Nordheim
#
# Question: Which treaty ended the Hundred-Year Peace in the ancient kingdom of Avalon?
# Candidates:
#
# 	1.	Treaty of Eldenbrooke
# 	2.	Treaty of Greendale
# 	3.	Treaty of Stormwind
# 	4.	Treaty of Blackwater
# The answer to this question: 2.	Treaty of Greendale
#
# Question: During the reign of which ruler did the mysterious “Age of Darkness” occur in the empire of Astoria?
# Candidates:
#
# 	1.	Emperor Midas the Golden
# 	2.	Emperor Pharos the Seer
# 	3.	Empress Cyra the Shadowed
# 	4.	King Alaric the Bold
# The answer to this question: 3.	Empress Cyra the Shadowed
#
# Question: The Great Flood of 1512 BCE in the land of Arcadia is said to have been caused by which deity, according to myth?
# Candidates:
#
# 	1.	Zephyrus, God of the Winds
# 	2.	Solara, Goddess of the Sun
# 	3.	Aethra, Goddess of Rain
# 	4.	Lycos, God of the Seas
# The answer to this question: 4.	Lycos, God of the Seas"""
        problem_string = str(problem)
        prompt = self.prompt_template.format(target_problem=problem_string,
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

            return relevant_problem, similarity
        except Exception as e:
            print(e)
            return None, None