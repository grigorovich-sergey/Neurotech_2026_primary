"""Instance 5 end-to-end experiment integration and scientific verification."""

from integration.analysis import generate_analysis
from integration.orchestrator import IntegratedExperimentOrchestrator
from integration.workflow import run_integrated_experiment

__all__ = (
    "IntegratedExperimentOrchestrator",
    "generate_analysis",
    "run_integrated_experiment",
)
