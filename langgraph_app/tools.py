from langchain_core.tools import tool
from typing import Annotated



def route_to_specialist(specialist:list[str]):
    
    @tool
    def route_to_specialist(specialist: Annotated[str, f"The name of the medical specialty among {specialist}"], reason: Annotated[str, "Why this specialist is needed."]) -> str:
        """Routes the patient to a specialist.
        Returns:
            A message indicating the routing decision.
        """
        print(f"Routing to {specialist} because: {reason}")
        return f"Routing to {specialist} because: {reason}"
    return route_to_specialist