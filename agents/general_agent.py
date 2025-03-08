from langchain_google_genai import ChatGoogleGenerativeAI # Import the Gemini model
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
load_dotenv()

#Use the api key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

#llm = ChatAnthropic(model="claude-3-5-sonnet-latest") #Old LLM Model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True) # Replace Anthropic with Gemini
SPECIALISTS = []


def set_specialist_prompt(specialist):
    SPECIALISTS.append(specialist)



def create_general_agent(tools: list[Tool]): #Pass tools as a parameter
    """Creates a ReAct agent for the general doctor."""
    GENERAL_AGENT_PROMPT = """
You are a general medical assistant.  Assess the patient's symptoms and give it to the correct specialist  if you dont have enough info. If specialist needed, say which  one among (dermatology,gastroentology,pediatrics) and WHY using the tool provided and END it . If no specialist is needed give a recommendation based on your general knowledge. If you think you have fully addressed the problem or task, respond with FINISH and a description of why.
"""
    print(GENERAL_AGENT_PROMPT)
    agent = create_react_agent(
        llm,
        tools,
        prompt=GENERAL_AGENT_PROMPT
    )
    print(agent)
    return agent