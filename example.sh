#!/bin/bash
cd /workspace/schema_instructor/refactored_sa-icl
# 1. Define arrays for all your changing variables
SCHEMA_TEMPLATES=("LeastToMostSchema" "StepBack" "ChemistrySchema2" "FactSchema2")
SOLVER_TYPES=("ComprehensiveSchemaSolver" "OneShotSolver")
SOLVER_MODELS=("deepseek" "Gemini")
SUBJECTS=("GPQA" "GPQAPhysics" "GPQABiology")

# 2. Define your static variables
TARGET_ITERATIONS=1
KNOWLEDGE_LEVEL="Similar"
START_SIMILARITY="0.10"
END_SIMILARITY="1.0"
RANDOM_FROM_K="False"
NUM_SHOTS=1
TOP_K=1
REATTEMPTS=3
EXCLUDE="''"
USE_SELF_CONSISTENCY="False"
INCLUDE_ANSWER_IN_EXAMPLE="True"
MEMORY_MODEL="Deepseek"

# Experiment counter
count=1

# 3. Iterate through all combinations using nested loops
for subject in "${SUBJECTS[@]}"; do
    for schema in "${SCHEMA_TEMPLATES[@]}"; do
        for solver in "${SOLVER_TYPES[@]}"; do
            for model in "${SOLVER_MODELS[@]}"; do

                # Conditional check for example_reasoning
                if [ "$solver" == "ComprehensiveSchemaSolver" ]; then
                    example_reasoning="False"
                else
                    # Assuming True for OneShotSolver; adjust if your script expects otherwise
                    example_reasoning="True"
                fi

                echo "Running Experiment $count of 48..."
                echo "Configuration: Subject=$subject | Schema=$schema | Solver=$solver | Model=$model"

                # Construct the arguments string cleanly
                ARGS="--target_iterations $TARGET_ITERATIONS \
--subject $subject \
--knowledge_level '$KNOWLEDGE_LEVEL' \
--solver_type $solver \
--solver_model $model \
--schema_template $schema \
--start_similarity $START_SIMILARITY \
--end_similarity $END_SIMILARITY \
--random_from_k $RANDOM_FROM_K \
--num_shots $NUM_SHOTS \
--top_k $TOP_K \
--example_reasoning $example_reasoning \
--reattempts $REATTEMPTS \
--exclude $EXCLUDE \
--use_self_consistency $USE_SELF_CONSISTENCY \
--include_answer_in_example=$INCLUDE_ANSWER_IN_EXAMPLE \
--memory_model=$MEMORY_MODEL"

                # Run python script
                eval python validation.py $ARGS

                ((count++))
            done
        done
    done
done

echo "All 48 experiments completed!"
