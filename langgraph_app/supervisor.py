from typing import Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END
from langgraph.types import Command
from langgraph_app.graph_state import GraphState

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Options for the supervisor agent
OPTIONS = ["general_agent", "specialized_agent", "record_message"]

SYSTEM_PROMPT = """
You are a supervisor tasked with managing a medical consultation.
The available workers are:

- general_agent:  A general medical assistant who can assess initial symptoms, provide basic advice, and determine if a specialist is needed.
- specialized_agent: A specialist with expertise in a specific medical field (e.g., dentistry, orthopedics, neurology). Use this agent when the general agent indicates a need for specialized knowledge.

Given the patient's symptoms and the responses so far, determine which worker should act next.
When the consultation is complete and no further action is needed, respond with record_message.
"""


class RouterOutput(TypedDict):
    """Output parser for supervisor."""
    next: Literal[*OPTIONS]


def supervisor_node(state: GraphState) -> Command[Literal[*OPTIONS, "__end__"]]:
    """The supervisor node in the graph."""

    messages = state["messages"]

    # Check if the last message is a tool use or tool output message
    #if messages and isinstance(messages[-1], HumanMessage) and "Routing to" in messages[-1].content:
    #    # Extract the specialist from the tool's output (assuming it's the tool's response)
    #    tool_response = messages[-1].content
    #    specialist = tool_response.split("Routing to ")[1].split(" because")[0]
    #    return Command(goto="specialized_agent")  # Route directly to the specialist
    if state["specialization"] is not  None: #Pass the specialisation over
        return Command(goto="specialized_agent")

    response = llm.with_structured_output(RouterOutput).invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
    ] + state["messages"])

    next_node = response["next"]
    if next_node == "FINISH":
        return Command(goto=END)  # LangGraph uses "__end__" automatically
    else:
        return Command(goto=next_node)