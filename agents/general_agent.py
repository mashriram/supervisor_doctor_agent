from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI # Import the Gemini model
from langchain_core.tools import Tool
from langgraph_app.tools import route_to_specialist
from langchain.agents import  AgentExecutor
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import chain
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
load_dotenv()

#Use the api key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

#llm = ChatAnthropic(model="claude-3-5-sonnet-latest") #Old LLM Model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True) # Replace Anthropic with Gemini

GENERAL_AGENT_PROMPT = """
You are a general medical assistant.  Assess the patient's symptoms and determine if a specialist is needed. If specialist needed, say which and WHY using the tool provided. If no specialist is needed give a recommendation based on your general knowledge. If you think you have fully addressed the problem or task, respond with FINISH and a description of why.
"""

def create_general_agent(tools: list[Tool]): #Pass tools as a parameter
    """Creates a ReAct agent for the general doctor."""

    agent = create_react_agent(
        llm,
        tools,
        prompt=GENERAL_AGENT_PROMPT
    )
    return agent