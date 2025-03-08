from database.postgres_db import store_chat_history,get_chat_history
from langgraph_app.graph_state import GraphState
from langchain_google_genai import ChatGoogleGenerativeAI # Import the Gemini model
import os
from dotenv import load_dotenv
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from agents.specialized_agent import get_context
from langchain_core.tools import tool
load_dotenv()

#Use the api key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

#llm = ChatAnthropic(model="claude-3-5-sonnet-latest") #Old LLM Model
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True) # Replace Anthropic with Gemini
@tool
def save_conversation_history(state: GraphState):
    """Records messages to the database. after the end of each conversation
     Input: GraphState
             - Attributes:
                messages: A list of messages in the conversation.
                patient_id: The ID of the patient.
                name: The name of the speaker, or the tool that is providing information
                specialization: The medical specialization if route to a specialized agent
                
    Returns: None"""
    print("Recording message")
    patient_id = state['patient_id']
    for message in state['messages']:
        store_chat_history(patient_id, message.content, message.role)


@tool
def load_conversation_history(patient_id :str):
    """Loads messages from the database. after the end of each conversation
     Input: patient_id: The ID of the patient.
     Returns: List of messages from Database
     """
    messages = get_chat_history(patient_id=patient_id,number_of_items_to_retreive=5)
    return messages
     
                



def create_langgraph(state:GraphState):
    """Creates the LangGraph graph using supervisor agent."""
    print("Creating LangGraph")

    

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # Create specialized agents
    dermatology_tool = get_context("dermatology")
    gastroentology_tool = get_context("gastroentology")

    # def add(a: float, b: float) -> float:
    #     """Add two numbers."""
    #     return a + b

    # def multiply(a: float, b: float) -> float:
    #     """Multiply two numbers."""
    #     return a * b

    # def web_search(query: str) -> str:
    #     """Search the web for information."""
    #     return (
    #         "Here are the headcounts for each of the FAANG companies in 2024:\n"
    #         "1. **Facebook (Meta)**: 67,317 employees.\n"
    #         "2. **Apple**: 164,000 employees.\n"
    #         "3. **Amazon**: 1,551,000 employees.\n"
    #         "4. **Netflix**: 14,000 employees.\n"
    #         "5. **Google (Alphabet)**: 181,269 employees."
    #     )

    save_chat_history_agent = create_react_agent(
        model=model,
        tools=[save_conversation_history],
        name="save_chat_history",
        prompt=" you record messages to the database. after the end of each conversation",
        state_schema=GraphState
        
    )
    
    load_chat_history_agent = create_react_agent(
        model=model,
        tools=[load_conversation_history],
        name="load_chat_history",
        prompt="you get chat history from database at the start of the conversation",
        state_schema=GraphState
    )
    dermatology_agent = create_react_agent(
        model=model,
        tools=[dermatology_tool],
        name="dermatologist",
        prompt="You are a dermatological expert.Use the RAG to query the textbook and propely ground with facts. Always use one tool at a time.",
        state_schema=GraphState
    )

    
    gastroentology_agent = create_react_agent(
        model=model,
        tools=[gastroentology_tool],
        name="gastroentologist",
        prompt="You are a gastroentological expert.Use the RAG to query the textbook and propely ground with facts. Always use one tool at a time.",
        state_schema=GraphState
    )

    # Create supervisor workflow
    workflow = create_supervisor(
        [gastroentology_agent, dermatology_agent,save_chat_history_agent,load_chat_history_agent],
        model=model,
        prompt=(
            "You are a general physician handling along with a team consisting  a dermatology_agent and a gastroentology_agent "
            "For gastric issues, use gastroentology_agnet. "
            "For skin issues, use dermatology_agent."
            "If it is pertaining other domains give ypu own general answer"
            "Use the load_chat_history_agent to load history"
            f"If the history is Non e then use this state \n  {state}"
            "Save the conversation once its over using save_chat_history_agent "
        ),
        state_schema=GraphState,
        output_mode="full_history"
        
    )

    # Compile and run
    app = workflow.compile()
    
    return app