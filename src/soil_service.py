from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to MongoDB Atlas
def get_latest_soil_data():
    connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    if not connection_string:
        print("Warning: MONGODB_CONNECTION_STRING not found in .env")
        return None
        
    try:
        client = MongoClient(connection_string)
        db = client.agriculture
        collection = db.soil_data
    
        # Assuming there is a 'timestamp' field that records the document's creation time
        latest_document_cursor = collection.find().sort("created_at", -1).limit(1)
        
        # Get the latest document
        for doc in latest_document_cursor:
            return doc
            
        return None
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None
