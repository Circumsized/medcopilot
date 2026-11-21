from .safety_guard import MedicalSafetyGuard

__all__ = [
    "AgentGraph",
    "MedicalConsultationWorkflow",
    "MedicalSafetyGuard",
    "Session",
    "ConsultationState",
]


def __getattr__(name: str):
    if name == "AgentGraph":
        from .agent_graph import AgentGraph
        return AgentGraph
    if name == "MedicalConsultationWorkflow":
        from .orchestrator import MedicalConsultationWorkflow
        return MedicalConsultationWorkflow
    if name in ("Session", "ConsultationState"):
        from . import agent_graph
        return getattr(agent_graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

