import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def validate_data_sources():
    print("🔍 Validating Data Sources...")
    
    # Check lang_model.txt
    lang_model_path = "data/lang_model.txt"
    if os.path.exists(lang_model_path):
        with open(lang_model_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"✅ lang_model.txt: {len(lines)} lines")
    else:
        print(f"❌ lang_model.txt not found at {lang_model_path}")
        # Create empty file to avoid errors
        os.makedirs("data", exist_ok=True)
        open(lang_model_path, 'w').close()
        print("📁 Created empty lang_model.txt to avoid errors")

    # Check main MongoDB
    try:
        client = MongoClient(os.getenv("MONGODB_URL"))
        db = client["elf_owl_db"]
        collection = db["dataset"]
        count = collection.count_documents({})
        print(f"✅ Main MongoDB: {count} documents")
        
        # Check sample document structure
        sample = collection.find_one()
        if sample:
            print(f"📄 Sample document keys: {list(sample.keys())}")
        client.close()
    except Exception as e:
        print(f"❌ Main MongoDB error: {e}")

    # Check math training MongoDB
    try:
        client = MongoClient(os.getenv("MATHS_TRAINING"))
        db = client["dataset"]
        collection = db["NumCrunch"]
        count = collection.count_documents({})
        print(f"✅ Math Training MongoDB: {count} documents")
        
        # Check sample math document
        sample = collection.find_one()
        if sample:
            print(f"📊 Sample math document: {sample}")
            # Check data types
            for key, value in sample.items():
                print(f"   {key}: {type(value)} - {str(value)[:50]}...")
        client.close()
    except Exception as e:
        print(f"❌ Math Training MongoDB error: {e}")

if __name__ == "__main__":
    validate_data_sources()