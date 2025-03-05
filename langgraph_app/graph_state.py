from typing import TypedDict, List, Dict, Any

from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: A list of messages in the conversation.
        patient_id: The ID of the patient.
        name: The name of the speaker, or the tool that is providing information
        specialization: The medical specialization if route to a specialized agent
    """
    messages: List[BaseMessage]
    patient_id: str
    name: str
    specialization: str