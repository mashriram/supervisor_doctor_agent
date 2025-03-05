from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import chain
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from agents.general_agent import create_general_agent
from agents.specialized_agent import create_specialized_agent
from database.postgres_db import store_chat_history
from langgraph_app.graph_state import GraphState
from langgraph_app.supervisor import supervisor_node
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI # Import the Gemini model
from langgraph_app.tools import route_to_specialist
import os
from dotenv import load_dotenv
load_dotenv()

#Use the api key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

#llm = ChatAnthropic(model="claude-3-5-sonnet-latest") #Old LLM Model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True) # Replace Anthropic with Gemini

def record_message(state: GraphState):
    """Records messages to the database."""
    patient_id = state['patient_id']
    for message in state['messages']:
        store_chat_history(patient_id, message.content, message.role)
    return state

def initialize_conversation(state: GraphState):
    """Logs the initial patient message into chat history"""
    patient_id = state['patient_id']
    for message in state['messages']:
        store_chat_history(patient_id, message.content, 'user')
    return state

def general_agent_node(state: GraphState):
    """General agent node that routes to other specialists"""
    tools = [route_to_specialist] #Define the tools inside
    general_agent = create_general_agent(tools)
    input_message = state['messages'][-1].content

    result = general_agent.invoke({"input": input_message, "chat_history": state["messages"],"messages":state['messages']}) # No Messages are needed
    
    #Check the tool section of the output
    specialization = None
    if "Routing to" in result["output"] and "FINISH" not in result["output"]: #Force check
        specialization = result["output"].split("Routing to ")[1].split(" because")[0] #This is because the `general_agent` has a "routing tool"
    elif "FINISH" in result["output"]:
        specialization = "end" #Force the agent to "end" for edge cases
    else:
        specialization = None #If for some reason, the tool is not in result but no FINISH is detected, there is nothing for the tool to act, hence "None".

    return {
        "messages": [HumanMessage(content=result["output"], name="general_agent")],
        "specialization": specialization #Always none here. The tool determines the specialisation based on the response of the Agent
    }

def specialized_agent_node(state: GraphState):
    """Specialized agent node that performs RAG"""
    specialization = state['specialization']
    input_message = state['messages'][-1].content

    if specialization:
        specialized_agent = create_specialized_agent(specialization)
        response = specialized_agent.invoke({"input": input_message})
        return {"messages": [HumanMessage(content=response, name="specialized_agent")], "specialization": None} #Always set this back to none
    else:
        return {"messages": [HumanMessage(content="Error: No specialization specified.", name = "specialized_agent")], "specialization": None} #Return None too

def end_node(state):
    return state

def create_langgraph():
    """Creates the LangGraph graph using supervisor agent."""
    print("Creating LangGraph")

    builder = StateGraph(GraphState)

    #Nodes
    builder.add_node("initialize_conversation", initialize_conversation)
    builder.add_node("general_agent", general_agent_node)
    builder.add_node("specialized_agent", specialized_agent_node)
    builder.add_node("record_message", record_message)
    builder.add_node("supervisor", supervisor_node) #Supervisor
    builder.add_node("end", end_node) #Fix the node by passing function

    #Edges
    builder.set_entry_point("initialize_conversation")

    #Initial
    builder.add_edge("initialize_conversation", "general_agent")

    #Add supervisor
    builder.add_edge("general_agent", "supervisor")
    builder.add_edge("specialized_agent", "supervisor")
    
    #Add record_message to the flow before supervisor node - NEW ADDITION TO ADDRESS RECURSION ERROR
    builder.add_edge("record_message", "supervisor") #Point the record message to the supervisor

    # Routing to specialized agents
    CONDITIONAL_MAP = {
        "general_agent": "general_agent",
        "specialized_agent": "specialized_agent",
        "end": "end", #Make it default to end
    }

    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["specialization"] if state["specialization"] != None and state["specialization"] != "end" else "end",
        CONDITIONAL_MAP
    )

    builder.add_edge("record_message", "end") #Fix the final edge

    graph = builder.compile()
    print(graph.get_graph().draw_mermaid())
    return graph