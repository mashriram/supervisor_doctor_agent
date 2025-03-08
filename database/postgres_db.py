import os
import psycopg2
from dotenv import load_dotenv


load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DBNAME = os.getenv("POSTGRES_DBNAME")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

def connect_to_postgres():
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DBNAME,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        cursor = conn.cursor()
        print("Connected to PostgreSQL")
        return conn, cursor
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        if conn:
            conn.close()
        return None, None

def get_user_info(patient_id):
    conn, cursor = connect_to_postgres()
    if not conn or not cursor:
        return None
    try:
        cursor.execute("SELECT name, medication_history FROM patients WHERE patient_id = %s", (patient_id,))
        result = cursor.fetchone()
        if result:
            name, medication_history = result
            return {"name": name, "medication_history": medication_history}
        else:
            return None
    except psycopg2.Error as e:
        print(f"Error fetching user info: {e}")
        return None
    finally:
        conn.close()


def store_chat_history(patient_id, message, role):
    conn, cursor = connect_to_postgres()
    if not conn or not cursor:
        return

    try:
        cursor.execute(
            "INSERT INTO chat_history (patient_id, message, role, timestamp) VALUES (%s, %s, %s, NOW())",
            (patient_id, message, role)
        )
        conn.commit()
    except psycopg2.Error as e:
        print(f"Error storing chat history: {e}")
        conn.rollback()  # Rollback in case of error
    finally:
        conn.close()
        
def get_chat_history(patient_id,number_of_items_to_retreive:int):
    conn, cursor = connect_to_postgres()
    if not conn or not cursor:
        return

    try:
        cursor.execute(
            "SELECT meesage,role FROM chat_history WHERE patient_id = %s",
            (patient_id,)
        )
        result = cursor.fetchmany(number_of_items_to_retreive)
        if result:
            message,role = result
            return {'message':message,"role":role}
        else:
            return None
    except psycopg2.Error as e:
        print(f"Error fetching chat history: {e}")
    finally:
        conn.close()
    

def register_user(patient_id, name, medication_history):
    conn, cursor = connect_to_postgres()
    if not conn or not cursor:
        return False
    try:
        cursor.execute(
            "INSERT INTO patients (patient_id, name, medication_history) VALUES (%s, %s, %s)",
            (patient_id, name, medication_history)
        )
        conn.commit()
        return True
    except psycopg2.Error as e:
        print(f"Error registering user: {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


#Example Usage (For setting up table schema)
postgres_conn, postgres_cursor = connect_to_postgres()

if postgres_conn and postgres_cursor:
    try:
        postgres_cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255),
                medication_history TEXT
            );
        """)

        postgres_cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                patient_id VARCHAR(255),
                message TEXT,
                role VARCHAR(50),
                timestamp TIMESTAMP WITHOUT TIME ZONE,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            );
        """)
        postgres_conn.commit()

        print("Tables created successfully.")
    except psycopg2.Error as e:
        print(f"Error creating tables: {e}")
    finally:
        postgres_conn.close()