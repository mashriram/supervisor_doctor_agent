import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from database.milvus_db import create_rag_collection, get_milvus_collection
import numpy as np

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Or another suitable model


def load_and_embed_docs(specialization_name: str, data_dir: str = "./data"):
    """
    Loads PDF documents from a directory, splits them into chunks,
    embeds the chunks using Sentence Transformers, and stores them in Milvus,
    but only if the collection is empty.
    """
    collection = create_rag_collection(specialization_name)

    # Check if the collection is empty
    if collection.num_entities > 0:
        print(f"Collection {specialization_name}_rag_collection is not empty. Skipping loading and embedding.")
        return True  # Indicate success (already loaded)

    embedding_model = SentenceTransformer(_EMBEDDING_MODEL) # Load SentenceTransformer model

    documents = []
    for filename in os.listdir(os.path.join(data_dir, specialization_name)):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, specialization_name, filename))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embed the chunks and store them in Milvus
    batch_size = 32 # Adjust batch size as needed

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk.page_content for chunk in batch]
        embeddings = embedding_model.encode(texts).tolist()

        # Prepare the data for insertion
        rows = []
        for i in range(len(texts)):
          rows.append({
              "embedding": embeddings[i],
              "text": texts[i]
          })

        try:
            # Insert the data
            collection.insert(rows)
            collection.flush()
            print(f"Inserted batch {i//batch_size + 1} of {len(chunks)//batch_size + 1} into Milvus")
        except Exception as e:
            print(f"Failed to insert data into Milvus: {e}")
            return False

    print(f"Successfully loaded and embedded documents for {specialization_name}")
    return True



def retrieve_context(specialization_name: str, query: str, top_k: int = 5):
    """
    Retrieves the top_k most relevant documents from Milvus for a given query.
    """
    collection = get_milvus_collection(f"{specialization_name}_rag_collection")
    if not collection:
        print(f"Collection {specialization_name}_rag_collection not found.")
        return []
    print(collection,"collection found ")
    embedding_model = SentenceTransformer(_EMBEDDING_MODEL)
    query_embedding = embedding_model.encode(query).tolist()

    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 64}  # Adjust ef for performance
    }

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )
    print("Results",results[0])
    for hits in results:
        for hit in hits:
            print("In hit")
            context = hit.entity.get("text")
            print(context)
    print("Context",context)
    print("Sending context")
    return context

def main():
    print("in Main")
    print(retrieve_context("dermatology","skin disease"))
    
if __name__ == "__main__":
    main()