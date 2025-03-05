import os
from pymilvus import connections, Collection, utility, MilvusException
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")

_VECTORS_DIMENSION = 384 # Example, adjust based on your embedding model
FACE_COLLECTION_NAME = "face_collection"

def connect_to_milvus():
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        print("Successfully connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False
    return True

def create_face_collection():
    from pymilvus import FieldSchema, CollectionSchema, DataType
    if utility.has_collection(FACE_COLLECTION_NAME):
        print(f"Collection {FACE_COLLECTION_NAME} already exists.  Loading...")
        return Collection(FACE_COLLECTION_NAME)

    fields = [
        FieldSchema(name="patient_id", dtype=DataType.VARCHAR, is_primary=True, max_length=200),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=_VECTORS_DIMENSION)
    ]
    schema = CollectionSchema(fields, "Face embeddings of patients")
    face_collection = Collection(FACE_COLLECTION_NAME, schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    face_collection.create_index("embedding", index_params)
    face_collection.load()  # Load into memory for faster searching

    print(f"Collection {FACE_COLLECTION_NAME} created successfully.")
    return face_collection


def create_rag_collection(specialization_name: str):
    from pymilvus import FieldSchema, CollectionSchema, DataType
    collection_name = f"{specialization_name}_rag_collection"

    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} already exists. Loading...")
        return Collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=_VECTORS_DIMENSION),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535) # Maximum length for text
    ]

    schema = CollectionSchema(fields, f"RAG data for {specialization_name}")
    collection = Collection(collection_name, schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index("embedding", index_params)
    collection.load()  # Load into memory
    print(f"Collection {collection_name} created successfully.")
    return collection



# Initialize on import
connect_to_milvus()
face_collection = create_face_collection()



def get_milvus_collection(collection_name: str):
    """
    Retrieves a Milvus collection by name.
    """
    try:
        return Collection(collection_name)
    except MilvusException as e:
        print(f"Error retrieving collection {collection_name}: {e}")
        return None