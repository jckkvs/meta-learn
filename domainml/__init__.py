"""
DomainML - Domain Knowledge Integrated Machine Learning Library
"""
__version__ = "0.3.0"

# Core
from domainml.core.metadata import (
    FeatureMetadata,
    ConstraintStrength,
    MonotonicityDirection,
    MonotonicityDirection,
    ManifoldConfig,
    ManifoldGroupConfig,
    monotonicity_to_direction,
    direction_to_monotonicity,
)
from domainml.core.pipeline import MetaPipeline

# Models
from domainml.models.base import DomainEstimator
from domainml.models.wrappers import MonotonicityWrapper

# Constraints
from domainml.constraints.monotonicity import MonotonicLinearRegression, generate_extrapolation_points
from domainml.constraints.engine import MonotonicityEngine
from domainml.constraints.kernel import KernelMonotonicity
from domainml.constraints.group import GroupConstraintEngine, GroupStandardScaler
from domainml.constraints.manifold_engine import ManifoldAssumptionEngine
from domainml.constraints.manifold_regularizer import ManifoldRegularizer, ManifoldPreprocessor

# Analysis
from domainml.analysis.metrics import satisfaction_score
from domainml.analysis.coef_checker import LinearCoefConflictChecker
from domainml.model_selection.cv import constrained_cv

__all__ = [
    # Core
    "FeatureMetadata",
    "ConstraintStrength",
    "MonotonicityDirection",
    "ManifoldConfig",
    "ManifoldGroupConfig",
    "monotonicity_to_direction",
    "direction_to_monotonicity",
    "MetaPipeline",
    # Models
    "DomainEstimator",
    "MonotonicityWrapper",
    # Constraints
    "MonotonicLinearRegression",
    "generate_extrapolation_points",
    "MonotonicityEngine",
    "KernelMonotonicity",
    "GroupConstraintEngine",
    "GroupStandardScaler",
    "ManifoldAssumptionEngine",
    "ManifoldRegularizer",
    "ManifoldPreprocessor",
    # Analysis
    "satisfaction_score",
    "LinearCoefConflictChecker",
    "constrained_cv",
]
