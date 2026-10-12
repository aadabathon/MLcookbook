"""Core abstractions: Dataset, schema/type inference, result serialization."""

from mlcookbook.core.dataset import ColumnSummary, Dataset, DatasetSummary
from mlcookbook.core.quality import QualityIssue, column_issues
from mlcookbook.core.results import Result, to_jsonable
from mlcookbook.core.schema import ColumnSchema, DatasetSchema, VarType, infer_var_type

__all__ = [
    "ColumnSchema",
    "ColumnSummary",
    "Dataset",
    "DatasetSchema",
    "DatasetSummary",
    "QualityIssue",
    "Result",
    "VarType",
    "column_issues",
    "infer_var_type",
    "to_jsonable",
]
