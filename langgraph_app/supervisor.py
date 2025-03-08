# from typing import Literal, TypedDict

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langgraph.graph import END
# from langgraph.types import Command
# from langgraph_app.graph_state import GraphState
# from pydantic import BaseModel

# # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


# # SPECIALISTS = []

# # Options for the supervisor agent
# # OPTIONS = ["general_agent", "dermatology",'gastroentology','pediatrics', "record_message"]

# # SYSTEM_PROMPT = f"""
# # You are a supervisor tasked with managing a medical consultation.
# # The available workers are:

# # - general_agent:  A general medical assistant who can assess initial symptoms, provide basic advice, and determine if a specialist is needed.
# # - dermatology: A specialist with expertise in a  dermatology among (dermatology,gastroentology,pediatrics) . Use this agent when the general agent indicates a need for dermatological knowledge.
# # - gastroentology:A specialist with expertise in a  gastroentology among (dermatology,gastroentology,pediatrics) . Use this agent when the general agent indicates a need for gastroentological knowledge.
# # - pediatrics: A specialist with expertise in a specific pediatrics among (dermatology,gastroentology,pediatrics) . Use this agent when the general agent indicates a need for pediatrical knowledge.

# # Given the patient's symptoms and the responses so far, determine which worker should act next.and FINISH
# # When the consultation is complete and no further action is needed, respond with FINISH.

# # The below is the general agents assesment \n
# # """


