from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from pymongo.mongo_client import MongoClient

# Define constants
DB_NAME = "logs_database"
COLLECTION_NAME = "pdf_collection"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
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
        for doc in docs:

            # Inspect the structure of the document
            #print(doc)

            text_content = doc.page_content  # Ensure the correct attribute for text content
            embedding = embeddings._embed(text_content)  # Generate embeddings

            # Create a dictionary representation of the document and add the embedding
            doc_dict = doc.__dict__.copy()
            doc_dict['embedding'] = embedding  # Add embedding to the dictionary


            collection.insert_one(doc.__dict__)  # Store document in MongoDB

# Store documents and embeddings in MongoDB
store_documents_and_embeddings(docs, MONGODB_COLLECTION, EMBEDDINGS)

# Initialize MongoDBAtlasVectorSearch
vector_search = MongoDBAtlasVectorSearch(
    mongodb_client=client,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)

# Perform similarity search
query = "What were the compute requirements for training GPT 4"
query_embedding = EMBEDDINGS._embed(query)
results = vector_search.vectorstore.similarity_search(query_embedding)
print("result: ", results)
