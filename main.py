import os
from face_recognition.face_rec import register_patient_face, scan_face_and_get_patient_id
from database.postgres_db import get_user_info, connect_to_postgres, register_user
from langgraph_app.langgraph_flow import create_langgraph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langgraph_app.graph_state import GraphState
from rag.rag_utils import load_and_embed_docs
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def initialize_state(patient_id: str, patient_info: str):
    messages = [HumanMessage(content=patient_info)]

    state = GraphState(
        messages=messages,
        patient_id=patient_id,
        name="Patient",
        specialization = None # Added specialization
    )
    return state

def main():
    # Connect to databases
    postgres_conn, postgres_cursor = connect_to_postgres()

    # Load RAG Data
    load_and_embed_docs("dermatology")
    load_and_embed_docs("gastroentology")
    load_and_embed_docs("pediatrics")

    # Create LangGraph
    graph = create_langgraph()
    chain = graph

    # Simulate the patient interaction
    face_image_path = "patient_image.jpg"  # Replace with actual path

    # **Register the test patient's face:**
    register_patient_face("test_patient", face_image_path)  # Uncomment to register a new face

    # Identify face from camera
    recognized_patient_id = scan_face_and_get_patient_id(face_image_path)

    if recognized_patient_id:
        user_info = get_user_info(recognized_patient_id)
        if user_info:
            medication_history = user_info["medication_history"]
            initial_patient_information = f"Patient presents with a headache and fatigue. Current medications: {medication_history if medication_history else 'None'}."
            print(f"Found the patient from the database.")
        else:
            print(f"Patient with id {recognized_patient_id} has not been registered. Please add a name and medical history")
            medication_history = "Unknown"
            initial_patient_information = f"Patient presents with a headache and fatigue. Current medications: {medication_history if medication_history else 'None'}."
            name = input("Please enter the patient's name: ")
            medication_history = input("Please enter the patient's medication history: ")
            register_user(recognized_patient_id, name, medication_history)

        # Initialize the state
        state = initialize_state(recognized_patient_id, initial_patient_information)
        print("State Init",state)

        #Run the LangGraph
        result = chain.invoke(state,{"recursion_limit": 3})

        # Print the final result
        print(result)
    else:
        print("Could not identify patient.")

    # Cleanup
    if postgres_conn:
        postgres_conn.close()

if __name__ == "__main__":
    main()