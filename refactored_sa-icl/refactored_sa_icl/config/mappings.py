from __future__ import annotations

"""
Static mappings used to build ExperimentSettings from CLI arguments.

These are copied from the legacy top-level config.py but namespaced for
the refactored pipeline.
"""

from typing import Dict, Any


subject_to_config: Dict[str, Dict[str, Any]] = {
    "GPQA": {
        "dataset_names": [["GPQA", "Synthetic"]],
        "problem_names_sizes": [["GPQA", 96]],
        "mapping_path": "mappings/rerank_mapping_chemistry.json",
    },
    "GPQAPhysics": {
        "dataset_names": [["GPQAPhysics", "SyntheticPhysics"]],
        "problem_names_sizes": [["GPQAPhysics", 100]],
        "mapping_path": "mappings/rerank_mapping_physics.json",
    },
    "GPQABiology": {
        "dataset_names": [["GPQABiology", "SyntheticBiology"]],
        "problem_names_sizes": [["GPQABiology", 100]],
        "mapping_path": "mappings/rerank_mapping_biology.json",
    },
    "MedXpertQA_Skeletal": {
        "dataset_names": [["MedXpertQA_Skeletal", "SyntheticMedXpertQA_Skeletal"]],
        "problem_names_sizes": [["MedXpertQA_Skeletal", 250]],
        "mapping_path": "mappings/rerank_mapping_medxpertqa.json",
    },
    "MMLUCollegeMath": {
        "dataset_names": [["MMLUCollegeMath", "MMLUCollegeMathSynthetic"]],
        "problem_names_sizes": [["MMLUCollegeMath", 100000]],
        "mapping_path": "mappings/rerank_mapping_mmlucollegemath.json",
    },
    "MMLUBusiness": {
        "dataset_names": [["MMLUBusiness", "MMLUBusinessSynthetic"]],
        "problem_names_sizes": [["MMLUBusiness", 100000]],
        "mapping_path": "mappings/rerank_mapping_mmlubusiness.json",
    },
    "CommonSenseQA": {
        "dataset_names": [["CommonSenseQA"]],
        "problem_names_sizes": [["CommonSenseQA", 100000]],
        "mapping_path": "mappings/rerank_mapping_commonsenseqa.json",
    },

    "Vision": {
        "dataset_names": [["Vision", "VisionSynthetic"]],
        "problem_names_sizes": [["Vision", 100000]],
        "mapping_path": "mappings/rerank_mapping_vision.json",
    },

    "Safety": {
        "dataset_names": [["SafetySynthetic"]],
        "problem_names_sizes": [["Safety", 100000]],
        "mapping_path": "mappings/rerank_mapping_safety.json",
    },
    "MMLUProStat": {
        "dataset_names": [["MMLUProStat", "MMLUProStatSynthetic"]],
        "problem_names_sizes": [["MMLUProStat", 100000]],
        "mapping_path": "mappings/rerank_mapping_mmluprost.json",
    },
    "MMLUProFinance": {
        "dataset_names": [["MMLUProFinance", "MMLUProFinanceSynthetic"]],
        "problem_names_sizes": [["MMLUProFinance", 100000]],
        "mapping_path": "mappings/rerank_mapping_mmluprof.json",
    },

    "GPQANoSynthetic": {
        "dataset_names": [["GPQA"]],
        "problem_names_sizes": [["GPQA", 96]],
        "mapping_path": "mappings/rerank_mapping_chemistry_no_synthetic.json",
    },

    "GPQAWithSynthetic": {
        "dataset_names": [["GPQA", "Synthetic"]],
        "problem_names_sizes": [["GPQA", 96]],
        "mapping_path": "mappings/rerank_mapping_chemistry_with_synthetic.json",
    },
}


knowledge_level_to_config: Dict[str, Dict[str, Any]] = {
    "Essentially Same": {"retreiver_approach": "paraphrase"},
    "Similar": {"retreiver_approach": "new_question"},
    "Different": {"retreiver_approach": "new_question_exam"},
    "rag_rerank": {
        "retreiver_approach": "rag_rerank",
    },
    "internal": {
        "retreiver_approach": "internal"
    },
    "history": {
        "retreiver_approach": "history"
    }
}


solver_type_to_config: Dict[str, Dict[str, Any]] = {
    "Baseline": {"solver_type": "BaselineSolver"},
    "Schema Only": {"solver_type": "SchemaSolver"},
    "NewExampleSchemaSolver": {"solver_type": "NewExampleSchemaSolver"},
    "One-Shot": {"solver_type": "ExampleSolver"},
    "One-Shot + Schema": {"solver_type": "ExampleSchemaSolver"},
    "ExampleSchemaSolver": {"solver_type": "ExampleSchemaSolver"},
    "Example Schema Only": {"solver_type": "ExampleSchemaNoActivationSolver"},
    "3-Shot": {"solver_type": "ExampleSolver", "top_k": 3, "num_shots": 3},
    "5-Shot": {"solver_type": "ExampleSolver", "top_k": 5, "num_shots": 5},
    "3-Shot + Schema": {
        "solver_type": "ExampleSchemaSolver",
        "top_k": 3,
        "num_shots": 3,
    },
    "5-Shot + Schema": {
        "solver_type": "ExampleSchemaSolver",
        "top_k": 5,
        "num_shots": 5,
    },

    "NewSchemaSolver": {"solver_type": "NewSchemaSolver"},
    "OneShotSchemaSolver": {"solver_type": "OneShotSchemaSolver"},
    "OneShotSolver": {"solver_type": "OneShotSolver"},
    "ZeroShotSchemaSolver": {"solver_type": "ZeroShotSchemaSolver"},
    "ComprehensiveSchemaSolver": {"solver_type": "ComprehensiveSchemaSolver"},
    "CoTSolver": {"solver_type": "CoTSolver"}
}


solver_model_to_config: Dict[str, str] = {
    "GPT-4o Mini": "GPT4oMini",
    "Claude": "Claude",
    "GPT-4o": "GPT4o",
    "GPT-5": "GPT5",
    "Gemini": "Gemini",
    "Llama-3.1": "Llama3",
    "Ministral": "Ministral",
    "MistralSmall": "MistralSmall",
    "Qwen-3": "Qwen3",
    "deepseek": "Deepseek",
    "o3mini": "o3mini"
}
