import time
import gradio as gr

from llama_index.core.llms import ChatMessage,MessageRole

#Data Loaders
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.github import GithubClient,GithubRepositoryReader
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
#Indices and Storage
import pymongo
from llama_index.storage.kvstore.mongodb import MongoDBKVStore as MongoDBCache
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
#Pipeline
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
#Vector Embedding Model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel




LOGS_DIR = '/home/ubuntu/projects/ialoganalysis/logai-python/store/logs'
MONGODB_URL='mongodb+srv://loganalysis:Dn750102@loganalysismongodb.qvevx.mongodb.net'
MONGODB_DBNAME='logs_database'
MONGODB_CLIENT       = pymongo.MongoClient(MONGODB_URL)
MONGODB_CACHE        = IngestionCache(cache=MongoDBCache(mongo_client=MONGODB_CLIENT, db_name = MONGODB_DBNAME))
MONGODB_DOCSTORE     = MongoDocumentStore.from_uri(uri=MONGODB_URL, db_name = MONGODB_DBNAME)


EMBED_MODEL          = 'BAAI/bge-small-en-v1.5'
EMBEDDINGS           = HuggingFaceEmbedding(model_name=EMBED_MODEL)

def chat_with_llm(message,history,radio,model):
 i               = len(message)
 query           = message[: i+1]
 response_str    = ''
 stream_response = get_stream_response(query,radio,model)
 if radio == 'none' :
   result = stream_response
   for r in result:
     time.sleep(0.1)
     response_str += r.delta
   yield response_str
 else:
   result = stream_response.response_gen
   for r in result:
     time.sleep(0.1)
     response_str += r
   yield response_str
   
search_index_model = SearchIndexModel(
  definition={
              "fields": [
                {
                   "type": "vector",
                   "numDimensions": 1536,
                   "path": "embedding",
                   "similarity": "euclidean | cosine | dotProduct"
                },
              ]
            },
            name="logs_idx",
            type="vectorSearch",
)

def ingest_logs():
    print('-> Ingest Logs')
    
    start = time.time()
    
    # Initialize sentence splitter
    splitter = SentenceSplitter(chunk_size=180, chunk_overlap=20)
    
    # Load data from the logs directory
    documents = SimpleDirectoryReader(LOGS_DIR, filename_as_id=True).load_data()
    
    # Initialize the MongoDB vector search
    '''
    vector_search = MongoDBAtlasVectorSearch(
        mongodb_client=MONGODB_CLIENT,
        db_name=MONGODB_DBNAME,
        collection_name='logs_full_collection',
        index_name='logs_full_idx'
    )
    
    '''
    # Create vector index using PyMongo
    #create_vector_index(MONGODB_CLIENT, MONGODB_DBNAME, 'logs_collection', 'logs_idx', 'embedding', 128)


    client = MongoClient(MONGODB_URL)
    # Access your database and collection
    database = client["logs_database"]
    collection = database["logs_collection"]

    vector_search = collection.create_search_index(model=search_index_model)
    #print("New search index named " + vector_search + " is building.")

    # Ensure the index is created before ingestion
    '''
    vector_search.create_index({
        'key': 'embedding',  # Replace with the actual field name
        'type': 'knnVector',
        'dimension': 128  # Ensure this matches your vector dimension
    })
    '''
    
    # Initialize the ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[splitter, EMBEDDINGS],
        vector_store=vector_search,
        cache=MONGODB_CACHE,
        docstore=MONGODB_DOCSTORE,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    
    # Run the pipeline
    nodes = pipeline.run(documents=documents)
    
    #vector_search = collection.create_search_index(model=search_index_model)
    print("New search index named " + vector_search + " is building.")

    end = time.time()
    
    print(f'  Total Time = {end - start}')
    print(f'  Total Documents = {len(documents)}')
    print(f'  Total Nodes = {len(nodes)}')

# Ensure you replace 'vector_field' with the actual field name in your MongoDB collection
# Ensure 'dimension' matches the dimension of the vectors you are using

ingest_logs()
