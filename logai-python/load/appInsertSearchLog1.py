import time
import numpy as np
import faiss
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pymongo.mongo_client import MongoClient
from uuid import uuid4  # To generate unique IDs

# Define constants
DB_NAME = "logs_database"
COLLECTION_NAME = "log_badwheather_collection"
VECTOR_INDEX_NAME = "vector_badwheather_index"
MONGODB_ATLAS_CLUSTER_URI = 'mongodb+srv://loganalysis:Dn750102@loganalysismongodb.qvevx.mongodb.net'
LOG_FILE_PATH = '/home/ubuntu/projects/ialoganalysis/logai-python/store/logs/full-bad-weather.log'

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# Initialize embedding model
EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
EMBEDDINGS = HuggingFaceEmbedding(model_name=EMBED_MODEL)

# Initialize MongoDB client and collection
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# Function to read log file and return documents
def read_log_file(file_path):
        with open(file_path, 'r') as file:
                    logs = file.readlines()
                        documents = [{'page_content': log.strip()} for log in logs]
                            return documents

                        # Function to store documents and embeddings in MongoDB
                        def store_documents_and_embeddings(docs, collection, embeddings):
                                metadata = []
                                    for doc in docs:
                                                # Access the correct attribute for text content
                                                        text_content = doc['page_content']
                                                                embedding = embeddings._embed(text_content)  # Use the correct method to generate embeddings

                                                                        # Debugging: Check if the embedding is generated correctly
                                                                                if not embedding:
                                                                                                print(f"Failed to generate embedding for text: {text_content}")
                                                                                                            continue

                                                                                                                # Create a dictionary representation of the document and add the embedding
                                                                                                                        doc_dict = {
                                                                                                                                            "_id": str(uuid4()),  # Unique ID for the document
                                                                                                                                                        "embedding": embedding,  # Directly use the embedding as it's already a list
                                                                                                                                                                    "text": text_content,
                                                                                                                                                                                "metadata": {
                                                                                                                                                                                                    "file_path": LOG_FILE_PATH,
                                                                                                                                                                                                                    "file_name": "full-bad-weather.log",
                                                                                                                                                                                                                                    "file_size": 4929,  # This should be dynamically generated if needed
                                                                                                                                                                                                                                                    "creation_date": datetime.now().strftime("%Y-%m-%d"),
                                                                                                                                                                                                                                                                    "last_modified_date": datetime.now().strftime("%Y-%m-%d")
                                                                                                                                                                                                                                                                                },
                                                                                                                                                                                            "_node_content": {
                                                                                                                                                                                                                "id_": str(uuid4()),  # Unique ID for the node content
                                                                                                                                                                                                                                "embedding": None,
                                                                                                                                                                                                                                                "metadata": None
                                                                                                                                                                                                                                                            },
                                                                                                                                                                                                        "_node_type": "TextNode",
                                                                                                                                                                                                                    "document_id": LOG_FILE_PATH,
                                                                                                                                                                                                                                "doc_id": LOG_FILE_PATH,
                                                                                                                                                                                                                                            "ref_doc_id": LOG_FILE_PATH
                                                                                                                                                                                                                                                    }

                                                                                                                                # Store document in MongoDB
                                                                                                                                        collection.insert_one(doc_dict)
                                                                                                                                                metadata.append(doc_dict['_id'])

                                                                                                                                                    # Ensure that the documents are inserted correctly and contain the 'embedding' field
                                                                                                                                                        inserted_docs = list(collection.find({}, {"embedding": 1}))
                                                                                                                                                            for doc in inserted_docs:
                                                                                                                                                                        if 'embedding' not in doc:
                                                                                                                                                                                        print(f"Document with _id {doc['_id']} does not contain 'embedding' field.")
                                                                                                                                                                                                    raise KeyError("The 'embedding' field is missing in one or more documents.")

                                                                                                                                                                                                    return metadata, np.array([doc['embedding'] for doc in inserted_docs]).astype('float32')

                                                                                                                                                                                                # Read log file
                                                                                                                                                                                                docs = read_log_file(LOG_FILE_PATH)

                                                                                                                                                                                                # Store documents and embeddings in MongoDB
                                                                                                                                                                                                metadata, embeddings_array = store_documents_and_embeddings(docs, MONGODB_COLLECTION, EMBEDDINGS)

                                                                                                                                                                                                # Initialize Faiss index
                                                                                                                                                                                                d = embeddings_array.shape[1]  # Dimension of the embeddings
                                                                                                                                                                                                index = faiss.IndexFlatL2(d)
                                                                                                                                                                                                index.add(embeddings_array)

                                                                                                                                                                                                # Function to perform similarity search
                                                                                                                                                                                                def faiss_similarity_search(query, top_k=5):
                                                                                                                                                                                                        query_embedding = np.array(EMBEDDINGS._embed(query)).astype('float32')  # Use the correct method to generate embeddings
                                                                                                                                                                                                            D, I = index.search(query_embedding.reshape(1, -1), top_k)
                                                                                                                                                                                                                results = []
                                                                                                                                                                                                                    for i in I[0]:
                                                                                                                                                                                                                                results.append(MONGODB_COLLECTION.find_one({'_id': metadata[i]}))
                                                                                                                                                                                                                                    return results

                                                                                                                                                                                                                                # Perform similarity search
                                                                                                                                                                                                                                query = "port 8081"
                                                                                                                                                                                                                                results = faiss_similarity_search(query)

                                                                                                                                                                                                                                # Print the log content of each result
                                                                                                                                                                                                                                for result in results:
                                                                                                                                                                                                                                        print("result: ", result['text'])
