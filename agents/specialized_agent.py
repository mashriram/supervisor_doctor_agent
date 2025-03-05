from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.rag_utils import retrieve_context
from langchain_core.tools import Tool
from langchain.agents import  AgentExecutor
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langgraph_app.tools import route_to_specialist
from langgraph.prebuilt import create_react_agent

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def create_specialized_agent(specialization_name: str):
    """
    Creates a specialized agent for a given medical specialization.
    """

    SPECIALIST_PROMPT = f"""
    You are a medical expert in {specialization_name}. Use the context provided to answer the patient's questions accurately and safely. If the answer is not in the context say you cannot answer it. If they ask for specific medication recommendations tell them to consult with the medical professional overseeing the case, and do not provide medical advice.

    Relevant context:
    {{context}}
    """


    def get_context(input):
        context = retrieve_context(specialization_name, input, top_k=3) # Retrieve context
        return "\n".join(context)


    def create_rag_react_agent(llm, specialization_name, context):
        """Creates a react agent for the RAG"""

        prompt = SPECIALIST_PROMPT.format(context = context) # Assign the tool

        agent = create_react_agent(
            llm,
            [], #No tools for the specialized agent
            prompt=prompt
        )
        return agent

    context = get_context("") #Pass an empty string since we are assigning the context inside
    agent = create_rag_react_agent(llm, specialization_name, context)

    return agent