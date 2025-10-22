from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

def fix_math_data_types():
    """Fix data type issues in math training data"""
    print("🔧 Fixing Math Training Data Types...")
    
    try:
        client = MongoClient(os.getenv("MATHS_TRAINING"))
        db = client["dataset"]
        collection = db["NumCrunch"]
        
        # Count documents with type issues
        total_docs = collection.count_documents({})
        print(f"📊 Total math documents: {total_docs}")
        
        # Find and fix documents with numeric values in string fields
        fixed_count = 0
        batch_updates = []
        
        for doc in collection.find({}):
            needs_update = False
            update_fields = {}
            
            # Check and fix input field
            if 'input' in doc and not isinstance(doc['input'], str):
                update_fields['input'] = str(doc['input'])
                needs_update = True
            
            # Check and fix output field  
            if 'output' in doc and not isinstance(doc['output'], str):
                update_fields['output'] = str(doc['output'])
                needs_update = True
            
            # Check and fix thinking field
            if 'thinking' in doc and not isinstance(doc['thinking'], str):
                update_fields['thinking'] = str(doc['thinking']) if doc['thinking'] is not None else "Solving mathematical problem"
                needs_update = True
            
            if needs_update:
                batch_updates.append({
                    'filter': {'_id': doc['_id']},
                    'update': {'$set': update_fields}
                })
                fixed_count += 1
                
                # Batch update every 1000 documents
                if len(batch_updates) >= 1000:
                    for update in batch_updates:
                        collection.update_one(update['filter'], update['update'])
                    batch_updates = []
                    print(f"✅ Fixed {fixed_count} documents so far...")
        
        # Process remaining batch
        if batch_updates:
            for update in batch_updates:
                collection.update_one(update['filter'], update['update'])
        
        print(f"🎉 Fixed data types for {fixed_count} math documents")
        
        # Verify fix
        sample = collection.find_one({})
        print(f"📋 Sample after fix: {sample}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error fixing math data: {e}")

if __name__ == "__main__":
    fix_math_data_types()