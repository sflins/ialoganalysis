from pymongo.mongo_client import MongoClient
# Connect to your Atlas deployment
uri = "mongodb+srv://loganalysis:Dn750102@loganalysismongodb.qvevx.mongodb.net"
client = MongoClient(uri)
# Access your database and collection
database = client["logs_database"]
collection = database["logs_collection"]
# Get a list of the collection's search indexes and print them
cursor = collection.list_search_indexes()
for index in cursor:
        print(index)
