"""
Centralised constants for raw dataset file paths used by the refactored
Schema-ICL pipeline.

Paths are expressed relative to the project root (the directory that
contains this ``refactored_sa-icl`` folder).
"""

# MMLU / College Mathematics
MMLU_COLLEGE_MATH_RAW: str = (
    "src/entity/datasets/raw_files/mmlu_pro_math_college_mathematics.csv"
)
MMLU_COLLEGE_MATH_SYNTHETIC_RAW: str = (
    "src/entity/datasets/raw_files/mmlu_pro_math_college_mathematics_synthetic.csv"
)

MMLU_BUSINESS_RAW: str = (
    "src/entity/datasets/raw_files/mmlu_pro_business.csv"
)

MMLU_BUSINESS_SYNTHETIC_RAW: str = (
    "src/entity/datasets/raw_files/mmlu_pro_business_synthetic.csv"
)

# MMLU-Pro subsets
MMLU_PRO_STAT_RAW: str = (
    "refactored_sa-icl/refactored_sa_icl/entity/datasets/raw_files/MMLUProStat.csv"
)
MMLU_PRO_STAT_SYNTHETIC_RAW: str = (
    "refactored_sa-icl/refactored_sa_icl/entity/datasets/raw_files/MMLUProStatSynthetic.csv"
)
MMLU_PRO_FINANCE_RAW: str = (
    "refactored_sa-icl/refactored_sa_icl/entity/datasets/raw_files/MMLUProTheoremQAFinance.csv"
)
MMLU_PRO_FINANCE_SYNTHETIC_RAW: str = (
    "refactored_sa-icl/refactored_sa_icl/entity/datasets/raw_files/MMLUProFinanceSynthetic.csv"
)

# GPQA / Physics / Chemistry and MedXpert could be added here as needed.
