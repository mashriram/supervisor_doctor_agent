from langchain_google_genai import ChatGoogleGenerativeAI
from rag.rag_utils import retrieve_context
from langchain_core.tools import tool

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def get_context(specialization_name:str):
    @tool
    def rag_context(input:str):
        """ Retrieves  content from the text book"""
        context = retrieve_context(specialization_name, input, top_k=3) # Retrieve context
        return "\n".join(context)
    return rag_context

def get_dummy():
    print("dummy")
    
