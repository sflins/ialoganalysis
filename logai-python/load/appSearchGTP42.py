import time
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pymongo.mongo_client import MongoClient

# Define constants
DB_NAME = "logs_database"
COLLECTION_NAME = "pdf_collection"
VECTOR_INDEX_NAME = "vector_index"
MONGODB_ATLAS_CLUSTER_URI = 'mongodb+srv://loganalysis:Dn750102@loganalysismongodb.qvevx.mongodb.net'

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# Load PDF data
loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
data = loader.load()
docs = text_splitter.split_documents(data)

# Initialize embedding model
EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
EMBEDDINGS = HuggingFaceEmbedding(model_name=EMBED_MODEL)

# Initialize MongoDB client and collection
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# Function to store documents and embeddings in MongoDB
def store_documents_and_embeddings(docs, collection, embeddings):
        metadata = []
        for doc in docs:
            # Inspect the structure of the document
            #print(doc)

            # Access the correct attribute for text content
            text_content = doc.page_content  # Assuming 'page_content' is the correct attribute
            embedding = embeddings._embed(text_content)  # Use the correct method to generate embeddings

            # Create a dictionary representation of the document and add the embedding
            doc_dict = doc.__dict__.copy()
            doc_dict['embedding'] = embedding  # Add embedding to the dictionary

            # Store document in MongoDB
            collection.insert_one(doc_dict)
            metadata.append(doc_dict['_id'])
                                                                                       
        return metadata, np.array([doc['embedding'] for doc in collection.find()]).astype('float32')

# Store documents and embeddings in MongoDB
metadata, embeddings_array = store_documents_and_embeddings(docs, MONGODB_COLLECTION, EMBEDDINGS)

# Initialize Faiss index
d = embeddings_array.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(d)
index.add(embeddings_array)

# Function to perform similarity search
def faiss_similarity_search(query, top_k=5):
    query_embedding = np.array(EMBEDDINGS._embed(query)).astype('float32')
    # Use the correct method to generate embeddings
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    results = []
    for i in I[0]:
        results.append(MONGODB_COLLECTION.find_one({'_id': metadata[i]}))
    return results

# Perform similarity search
query = "What were the compute requirements for training GPT 4"
results = faiss_similarity_search(query)
print("result: ", results)
