from langchain_core.tools import tool
from typing import Annotated

@tool
def route_to_specialist(specialist: Annotated[str, "The name of the medical specialty (e.g., dentist, ortho, brain)."], reason: Annotated[str, "Why this specialist is needed."]) -> str:
    """Routes the patient to a specialist.
    Returns:
        A message indicating the routing decision.
    """
    return f"Routing to {specialist} because: {reason}"