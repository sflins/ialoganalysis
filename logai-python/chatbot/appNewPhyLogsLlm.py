import time
import numpy as np
import faiss
import gradio as gr
import pymongo

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.storage.kvstore.mongodb import MongoDBKVStore as MongoDBCache
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

LOGS_DIR = '/home/ubuntu/projects/ialoganalysis/logai-python/store/logs'
MONGODB_URL = 'mongodb+srv://loganalysis:Dn750102@loganalysismongodb.qvevx.mongodb.net'
MONGODB_DBNAME = 'logs_database'
MONGODB_CLIENT = pymongo.MongoClient(MONGODB_URL)
MONGODB_CACHE = IngestionCache(cache=MongoDBCache(mongo_client=MONGODB_CLIENT, db_name=MONGODB_DBNAME))
MONGODB_DOCSTORE = MongoDocumentStore.from_uri(uri=MONGODB_URL, db_name=MONGODB_DBNAME)

EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
EMBEDDINGS = HuggingFaceEmbedding(model_name=EMBED_MODEL)

def chat_with_llm(message, history, radio, model):
    i = len(message)
    query = message[: i + 1]
    response_str = ''
    stream_response = get_stream_response(query, radio, model)
    if radio == 'none':
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

def ingest_logs():
    print('-> Ingest Logs')
    
    start = time.time()
    
    # Initialize sentence splitter
    splitter = SentenceSplitter(chunk_size=180, chunk_overlap=20)
    
    # Load data from the logs directory
    documents = SimpleDirectoryReader(LOGS_DIR, filename_as_id=True).load_data()
    
    # Extract embeddings and prepare for Faiss indexing
    embeddings = []
    metadata = []
    for doc in documents:
        embedding = EMBEDDINGS._embed(doc['text'])
        embeddings.append(embedding)
        metadata.append(doc['_id'])
    
    embeddings = np.array(embeddings).astype('float32')
    
    # Initialize Faiss index
    d = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    # Store metadata and embeddings in MongoDB
    db = MONGODB_CLIENT[MONGODB_DBNAME]
    collection = db['logs_full_collection']
    for doc, emb in zip(documents, embeddings):
        doc['embedding'] = emb.tolist()  # Store embedding as list
        collection.insert_one(doc)
    
    end = time.time()
    
    print(f'  Total Time = {end - start}')
    print(f'  Total Documents = {len(documents)}')

def search(query, top_k=5):
    query_embedding = EMBEDDINGS.transform(query).astype('float32')
    D, I = index.search(query_embedding, top_k)
    results = []
    for i in I[0]:
        results.append(collection.find_one({'_id': metadata[i]}))
    return results

if __name__ == "__main__":
    ingest_logs()
