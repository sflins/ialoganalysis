from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from pymongo.mongo_client import MongoClient


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
data = loader.load()
docs = text_splitter.split_documents(data)

DB_NAME = "logs_database"
COLLECTION_NAME = "pdf_collection"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
MONGODB_ATLAS_CLUSTER_URI = uri = 'mongodb+srv://loganalysis:Dn750102@loganalysismongodb.qvevx.mongodb.net'

EMBED_MODEL          = 'BAAI/bge-small-en-v1.5'
EMBEDDINGS           = HuggingFaceEmbedding(model_name=EMBED_MODEL)

client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

vector_search = MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
                embedding=EMBEDDINGS,
                    collection=MONGODB_COLLECTION,
                        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
                        )

query = "What were the compute requirements for training GPT 4"
results = vector_search.similarity_search(query)
print("result: ", results)
