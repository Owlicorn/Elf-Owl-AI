import os
import json
from typing import List, Dict, Any, Iterator, Tuple, Optional
from pymongo import MongoClient
import random
import time
from tqdm import tqdm
import numpy as np
from collections import defaultdict, deque
import threading
from datetime import datetime
import config

class DataLoader:
    def __init__(self, persistent_mongo: bool = False):
        self.config = config.Config()
        self._tokenizer = None
        self.persistent_mongo = persistent_mongo
        self._mongo_client = None
        self._maths_mongo_client = None
        self._context_cache = {}
        self._last_cache_clear = time.time()
        
        # For mixing streams
        self._lang_model_buffer = deque()
        self._mongo_buffer = deque()
        self._maths_buffer = deque()
        self._min_buffer_size = 100
        
        # Track loaded counts to enforce limits
        self._mongo_loaded_count = 0
        self._maths_loaded_count = 0
        self._lang_model_loaded_count = 0
        
        # Initialize connections if persistent mode
        if persistent_mongo:
            self._get_mongo_connection()
            self._get_maths_mongo_connection()

    def get_total_data_size_estimate(self) -> int:
        """Get estimated total data size from all sources WITH LIMITS APPLIED"""
        print("🔍 Estimating total data size with limits...")
        total = 0
        
        # Count lang_model lines WITH LIMIT
        try:
            if os.path.exists(self.config.LANG_MODEL_PATH):
                with open(self.config.LANG_MODEL_PATH, 'r', encoding='utf-8') as f:
                    lang_count = sum(1 for line in f if line.strip())
                # Apply limit
                lang_count = min(lang_count, self.config.MAX_LANG_MODEL_LINES)
                total += lang_count
                print(f"   📄 lang_model.txt: {lang_count} lines (limit: {self.config.MAX_LANG_MODEL_LINES})")
        except Exception as e:
            print(f"   ⚠️ Error counting lang_model: {e}")
        
        # Count MongoDB documents WITH LIMIT
        try:
            client = self._get_mongo_connection()
            if client:
                db = client[self.config.DATABASE_NAME]
                collection = db[self.config.COLLECTION_NAME]
                mongo_count = collection.count_documents({})
                # Apply limit
                mongo_count = min(mongo_count, self.config.MAX_MONGO_EXAMPLES)
                total += mongo_count
                print(f"   🗄️ Main MongoDB: {mongo_count} documents (limit: {self.config.MAX_MONGO_EXAMPLES})")
                self._close_mongo_connection(client)
        except Exception as e:
            print(f"   ⚠️ Error counting MongoDB: {e}")
        
        # Count Math Training documents WITH LIMIT
        try:
            maths_client = self._get_maths_mongo_connection()
            if maths_client:
                db = maths_client[self.config.MATHS_TRAINING_DB]
                collection = db[self.config.MATHS_TRAIN_COLLECTION]
                math_count = collection.count_documents({})
                # Apply limit
                math_count = min(math_count, self.config.MAX_MATHS_EXAMPLES)
                total += math_count
                print(f"   🔢 Math Training: {math_count} documents (limit: {self.config.MAX_MATHS_EXAMPLES})")
                self._close_maths_mongo_connection(maths_client)
        except Exception as e:
            print(f"   ⚠️ Error counting Math Training: {e}")
        
        print(f"📊 Total estimated data (with limits): {total:,} examples")
        return total
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for proper token counting and semantic search"""
        self._tokenizer = tokenizer
    
    def _get_mongo_connection(self) -> Optional[MongoClient]:
        """Get MongoDB connection with persistence option"""
        if self._mongo_client and self.persistent_mongo:
            try:
                self._mongo_client.admin.command('ping')
                return self._mongo_client
            except:
                self._mongo_client = None
        
        if not self.config.MONGODB_URL:
            return None
        
        try:
            client = MongoClient(
                self.config.MONGODB_URL,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=30000,
                maxPoolSize=20,
                minPoolSize=5 if self.persistent_mongo else 1,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=10000
            )
            
            client.admin.command('ping')
            
            if self.persistent_mongo:
                self._mongo_client = client
                print("🔗 Persistent MongoDB connection established")
            
            return client
            
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            return None

    def _get_maths_mongo_connection(self) -> Optional[MongoClient]:
        """Get Math Training MongoDB connection"""
        if self._maths_mongo_client and self.persistent_mongo:
            try:
                self._maths_mongo_client.admin.command('ping')
                return self._maths_mongo_client
            except:
                self._maths_mongo_client = None
        
        if not self.config.MATHS_TRAINING_URL:
            print("⚠️ Math training MongoDB URL not configured")
            return None
        
        try:
            client = MongoClient(
                self.config.MATHS_TRAINING_URL,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=30000,
                maxPoolSize=15,
                minPoolSize=3 if self.persistent_mongo else 1,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=10000
            )
            
            client.admin.command('ping')
            
            if self.persistent_mongo:
                self._maths_mongo_client = client
                print("🔗 Persistent Math Training MongoDB connection established")
            
            return client
            
        except Exception as e:
            print(f"❌ Math Training MongoDB connection failed: {e}")
            return None
    
    def _close_mongo_connection(self, client):
        """Close MongoDB connection"""
        if client and (client is not self._mongo_client or not self.persistent_mongo):
            try:
                client.close()
            except Exception as e:
                print(f"⚠️ Error closing MongoDB connection: {e}")
    
    def _close_maths_mongo_connection(self, client):
        """Close Math Training MongoDB connection"""
        if client and (client is not self._maths_mongo_client or not self.persistent_mongo):
            try:
                client.close()
            except Exception as e:
                print(f"⚠️ Error closing Math Training MongoDB connection: {e}")
    
    def _safe_math_data_processing(self, math_example):
        """Safely process math data with type handling"""
        try:
            # Safely extract and convert fields
            input_raw = math_example.get('input', '')
            output_raw = math_example.get('output', '') 
            thinking_raw = math_example.get('thinking', '')
            
            # Convert to string safely
            input_text = str(input_raw) if input_raw is not None else ""
            output_text = str(output_raw) if output_raw is not None else ""
            thinking_text = str(thinking_raw) if thinking_raw is not None else "Solving the mathematical problem step by step"
            
            # Clean and validate
            input_text = input_text.strip()
            output_text = output_text.strip()
            thinking_text = thinking_text.strip()
            
            # Validate content
            if (len(input_text) > 2 and len(output_text) > 1 and
                any(math_keyword in input_text.lower() for math_keyword in 
                    ['calculate', 'solve', 'math', 'equation', '=', '+', '-', '*', '/', 'x', 'y'])):
                
                return {
                    "input": input_text,
                    "thinking": thinking_text,
                    "output": output_text,
                    "mood": random.choice(["analytical", "precise", "logical", "educational"]),
                    "context_used": [],
                    "source": "math_training"
                }
        
        except Exception as e:
            print(f"⚠️ Error processing math example: {e}")
            if 'input' in math_example:
                print(f"   Problem input: {math_example['input']}")
        
        return None

    def _fill_buffers(self, target_size: int = 500):
        """Fill all three buffers for mixed streaming WITH LIMIT ENFORCEMENT"""
        # Fill lang_model buffer WITH LIMIT
        if len(self._lang_model_buffer) < target_size and self._lang_model_loaded_count < self.config.MAX_LANG_MODEL_LINES:
            try:
                if os.path.exists(self.config.LANG_MODEL_PATH):
                    remaining_lines = self.config.MAX_LANG_MODEL_LINES - self._lang_model_loaded_count
                    lines_to_load = min(target_size, remaining_lines)
                    
                    if lines_to_load > 0:
                        lang_stream = self.load_lang_model_data(max_lines=lines_to_load)
                        for text in lang_stream:
                            if text and len(text.strip()) > 10:  # Validate text
                                self._lang_model_buffer.append({
                                    "input": "Write a creative piece of text",
                                    "thinking": "I should generate creative text with good grammar and flow",
                                    "output": text.strip(),
                                    "mood": random.choice(["creative", "playful", "wise", "descriptive"]),
                                    "context_used": [],
                                    "source": "lang_model"
                                })
                                self._lang_model_loaded_count += 1
                                
                                if self._lang_model_loaded_count >= self.config.MAX_LANG_MODEL_LINES:
                                    print(f"📝 Reached lang_model limit: {self.config.MAX_LANG_MODEL_LINES}")
                                    break
                else:
                    print(f"⚠️ Lang model file not found: {self.config.LANG_MODEL_PATH}")
            except Exception as e:
                print(f"⚠️ Error filling lang_model buffer: {e}")

        # Fill MongoDB buffer WITH LIMIT
        if len(self._mongo_buffer) < target_size and self._mongo_loaded_count < self.config.MAX_MONGO_EXAMPLES:
            try:
                remaining_mongo = self.config.MAX_MONGO_EXAMPLES - self._mongo_loaded_count
                mongo_to_load = min(target_size, remaining_mongo)
                
                if mongo_to_load > 0:
                    mongo_stream = self.load_mongodb_conversations(limit=mongo_to_load)
                    for conv in mongo_stream:
                        input_text = conv.get('input') or conv.get('user_input') or conv.get('question', '')
                        output_text = conv.get('output') or conv.get('assistant_response') or conv.get('answer', '')
                        
                        # PROPER TYPE VALIDATION AND CONVERSION
                        input_text = str(input_text).strip() if input_text is not None else ""
                        output_text = str(output_text).strip() if output_text is not None else ""
                        thinking_text = str(conv.get('thinking', '')).strip() if conv.get('thinking') is not None else "Analyzing the user query and formulating a helpful response"

                        if (input_text and output_text and 
                            len(input_text) > 2 and len(output_text) > 2 and
                            len(input_text) < 1000 and len(output_text) < 1000):  # Reasonable length checks
                            
                            self._mongo_buffer.append({
                                "input": input_text,
                                "thinking": thinking_text,
                                "output": output_text,
                                "mood": conv.get('mood', random.choice(self.config.MOODS)),
                                "context_used": conv.get('context_used', []),
                                "source": "mongodb"
                            })
                            self._mongo_loaded_count += 1
                            
                            if self._mongo_loaded_count >= self.config.MAX_MONGO_EXAMPLES:
                                print(f"🗄️ Reached MongoDB limit: {self.config.MAX_MONGO_EXAMPLES}")
                                break
            except Exception as e:
                print(f"⚠️ Error filling MongoDB buffer: {e}")

        # Fill Math Training buffer WITH LIMIT
        if len(self._maths_buffer) < target_size and self._maths_loaded_count < self.config.MAX_MATHS_EXAMPLES:
            try:
                remaining_maths = self.config.MAX_MATHS_EXAMPLES - self._maths_loaded_count
                maths_to_load = min(target_size, remaining_maths)
                
                if maths_to_load > 0:
                    maths_stream = self.load_maths_training_data(limit=maths_to_load)
                    for math_example in maths_stream:
                        processed = self._safe_math_data_processing(math_example)
                        if processed:
                            self._maths_buffer.append(processed)
                            self._maths_loaded_count += 1
                            
                            if self._maths_loaded_count >= self.config.MAX_MATHS_EXAMPLES:
                                print(f"🔢 Reached Math Training limit: {self.config.MAX_MATHS_EXAMPLES}")
                                break
            except Exception as e:
                print(f"⚠️ Error filling Math Training buffer: {e}")

        return len(self._lang_model_buffer) > 0 or len(self._mongo_buffer) > 0 or len(self._maths_buffer) > 0
    
    def load_lang_model_data(self, max_lines: int = None) -> Iterator[str]:
        """Load raw text data for language modeling WITH LIMIT ENFORCEMENT"""
        if not os.path.exists(self.config.LANG_MODEL_PATH):
            print(f"❌ lang_model.txt not found at {self.config.LANG_MODEL_PATH}")
            return
        
        line_count = 0
        max_allowed = min(max_lines, self.config.MAX_LANG_MODEL_LINES) if max_lines else self.config.MAX_LANG_MODEL_LINES
        
        try:
            with open(self.config.LANG_MODEL_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
                        line_count += 1
                        if line_count >= max_allowed:
                            print(f"📝 Loaded {line_count} lines from lang_model (limit: {self.config.MAX_LANG_MODEL_LINES})")
                            break
        except Exception as e:
            print(f"❌ Error reading lang_model.txt: {e}")
    
    def load_mongodb_conversations(self, limit: int = None) -> Iterator[Dict]:
        """Load structured conversations from MongoDB WITH LIMIT ENFORCEMENT"""
        client = self._get_mongo_connection()
        if not client:
            return
        
        try:
            db = client[self.config.DATABASE_NAME]
            collection = db[self.config.COLLECTION_NAME]
            
            # Apply both the requested limit and the config limit
            effective_limit = min(limit, self.config.MAX_MONGO_EXAMPLES) if limit else self.config.MAX_MONGO_EXAMPLES
            
            cursor = collection.find({}).limit(effective_limit)
            
            loaded_count = 0
            for doc in cursor:
                yield doc
                loaded_count += 1
                
            print(f"🗄️ Loaded {loaded_count} documents from MongoDB (limit: {self.config.MAX_MONGO_EXAMPLES})")
                
        except Exception as e:
            print(f"❌ MongoDB streaming error: {e}")
        finally:
            self._close_mongo_connection(client)

    def load_maths_training_data(self, limit: int = None) -> Iterator[Dict]:
        """Load math training data from MongoDB WITH LIMIT ENFORCEMENT"""
        client = self._get_maths_mongo_connection()
        if not client:
            return
        
        try:
            db = client[self.config.MATHS_TRAINING_DB]
            collection = db[self.config.MATHS_TRAIN_COLLECTION]
            
            # Apply both the requested limit and the config limit
            effective_limit = min(limit, self.config.MAX_MATHS_EXAMPLES) if limit else self.config.MAX_MATHS_EXAMPLES
            
            cursor = collection.find({}).limit(effective_limit)
            
            loaded_count = 0
            for doc in cursor:
                yield doc
                loaded_count += 1
                
            print(f"🔢 Loaded {loaded_count} documents from Math Training (limit: {self.config.MAX_MATHS_EXAMPLES})")
                
        except Exception as e:
            print(f"❌ Math Training MongoDB streaming error: {e}")
        finally:
            self._close_maths_mongo_connection(client)
    
    def get_training_pairs_streaming(self, batch_size: int = 1000) -> Iterator[List[Dict]]:
        """Streaming generator that yields mixed batches from all sources WITH LIMITS"""
        # Reset counters for new streaming session
        self._mongo_loaded_count = 0
        self._maths_loaded_count = 0
        self._lang_model_loaded_count = 0
        
        print(f"🎯 Starting streaming with limits - MongoDB: {self.config.MAX_MONGO_EXAMPLES}, Math: {self.config.MAX_MATHS_EXAMPLES}, Lang: {self.config.MAX_LANG_MODEL_LINES}")
        
        while True:
            # Refill buffers if needed
            if (len(self._lang_model_buffer) < self._min_buffer_size and 
                len(self._mongo_buffer) < self._min_buffer_size and 
                len(self._maths_buffer) < self._min_buffer_size):
                if not self._fill_buffers(target_size=self._min_buffer_size * 2):
                    # No more data available or limits reached
                    if not self._lang_model_buffer and not self._mongo_buffer and not self._maths_buffer:
                        print("📭 All data sources exhausted or limits reached")
                        return

            batch: List[Dict] = []
            while len(batch) < batch_size:
                # Use round-robin sampling from all available sources
                sources = []
                if self._mongo_buffer:
                    sources.append(('mongodb', self._mongo_buffer))
                if self._lang_model_buffer:
                    sources.append(('lang_model', self._lang_model_buffer))
                if self._maths_buffer:
                    sources.append(('math_training', self._maths_buffer))
                
                if not sources:
                    break
                
                # Randomly select from available sources for diversity
                source_name, source_buffer = random.choice(sources)
                if source_buffer:
                    batch.append(source_buffer.popleft())

            if not batch:
                return

            yield batch
    
    def get_training_pairs(self, max_examples: int = None) -> List[Dict[str, Any]]:
        """Convert all data to training format - non-streaming mode WITH LIMITS"""
        print("📚 Loading training data with limits...")
        training_data: List[Dict[str, Any]] = []

        # Load lang_model data WITH LIMIT
        try:
            lang_texts = list(self.load_lang_model_data(max_lines=self.config.MAX_LANG_MODEL_LINES))
            print(f"📄 Loaded {len(lang_texts)} lines from lang_model.txt (limit: {self.config.MAX_LANG_MODEL_LINES})")
            for text in lang_texts:
                training_data.append({
                    "input": "Write a creative piece of text",
                    "thinking": "I should generate creative text with good grammar and flow",
                    "output": text,
                    "mood": random.choice(["creative", "playful", "wise", "descriptive"]),
                    "context_used": [],
                    "source": "lang_model"
                })
        except Exception as e:
            print(f"⚠️ Error loading lang_model data: {e}")

        # Load MongoDB conversations WITH LIMIT
        try:
            mongo_conversations = list(self.load_mongodb_conversations(limit=self.config.MAX_MONGO_EXAMPLES))
            print(f"🗄️ Loaded {len(mongo_conversations)} conversations from MongoDB (limit: {self.config.MAX_MONGO_EXAMPLES})")
            for conv in mongo_conversations:
                input_text = conv.get('input') or conv.get('user_input') or conv.get('question', '')
                output_text = conv.get('output') or conv.get('assistant_response') or conv.get('answer', '')

                if input_text and output_text and len(input_text.strip()) > 2 and len(output_text.strip()) > 2:
                    training_data.append({
                        "input": input_text.strip(),
                        "thinking": conv.get('thinking', 'Analyzing the user query and formulating a helpful response').strip(),
                        "output": output_text.strip(),
                        "mood": conv.get('mood', random.choice(self.config.MOODS)),
                        "context_used": conv.get('context_used', []),
                        "source": "mongodb"
                    })
        except Exception as e:
            print(f"⚠️ Error loading MongoDB data: {e}")

        # Load Math Training data WITH LIMIT
        try:
            maths_data = list(self.load_maths_training_data(limit=self.config.MAX_MATHS_EXAMPLES))
            print(f"🔢 Loaded {len(maths_data)} examples from Math Training DB (limit: {self.config.MAX_MATHS_EXAMPLES})")
            valid_math_count = 0
            for math_example in maths_data:
                processed = self._safe_math_data_processing(math_example)
                if processed:
                    training_data.append(processed)
                    valid_math_count += 1
            print(f"✅ Successfully processed {valid_math_count} math examples")
        except Exception as e:
            print(f"⚠️ Error loading Math Training data: {e}")

        # Shuffle training data
        if training_data:
            random.shuffle(training_data)

        # Trim to max_examples if specified (this is additional to source limits)
        if max_examples and len(training_data) > max_examples:
            training_data = training_data[:max_examples]
            print(f"✂️ Further limited to {max_examples} total examples")

        print(f"📚 Total training pairs: {len(training_data)}")

        # Show composition
        lang_count = len([d for d in training_data if d.get("source") == "lang_model"])
        mongo_count = len([d for d in training_data if d.get("source") == "mongodb"])
        maths_count = len([d for d in training_data if d.get("source") == "math_training"])
        print(f"📊 Composition: {lang_count} lang_model + {mongo_count} MongoDB + {maths_count} Math Training examples")

        return training_data
    
    def get_factual_context_for_inference(self, query: str, max_contexts: int = 2) -> List[Dict]:
        """Simple context retrieval for inference fallback"""
        contexts = []
        
        # Try to get from main MongoDB
        client = self._get_mongo_connection()
        if client:
            try:
                db = client[self.config.DATABASE_NAME]
                collection = db[self.config.COLLECTION_NAME]
                
                for doc in collection.find().limit(10):
                    score = self._keyword_similarity(query, doc)
                    if score > 0.1:
                        contexts.append({
                            'input': doc.get('input', ''),
                            'output': doc.get('output', ''),
                            'score': score
                        })
            except Exception as e:
                print(f"⚠️ Error in basic context retrieval: {e}")
            finally:
                self._close_mongo_connection(client)
        
        return contexts[:max_contexts]
    
    def get_smart_factual_context(self, query: str, max_contexts: int = 3) -> List[Dict]:
        """Smart context retrieval using semantic search from all sources"""
        if time.time() - self._last_cache_clear > 3600:
            self._context_cache.clear()
            self._last_cache_clear = time.time()
        
        cache_key = hash(query.lower().strip())
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        # Check if query is math-related
        math_keywords = ['calculate', 'solve', 'math', 'equation', 'formula', 'derivative', 'integral', 
                        'algebra', 'geometry', 'trigonometry', 'calculus', 'statistics', 'probability']
        is_math_query = any(keyword in query.lower() for keyword in math_keywords)
        
        contexts = []
        
        # Get contexts from main MongoDB
        client = self._get_mongo_connection()
        if client:
            try:
                db = client[self.config.DATABASE_NAME]
                collection = db[self.config.COLLECTION_NAME]
                
                for doc in collection.find().limit(100):
                    score = self._semantic_similarity(query, doc)
                    if score > 0.05:
                        contexts.append({
                            'score': score,
                            'input': doc.get('input', ''),
                            'output': doc.get('output', ''),
                            'thinking': doc.get('thinking', ''),
                            'mood': doc.get('mood', 'neutral'),
                            'similarity_type': 'semantic',
                            'source': 'main_mongodb'
                        })
            except Exception as e:
                print(f"❌ Error in main MongoDB context retrieval: {e}")
            finally:
                self._close_mongo_connection(client)
        
        # Get contexts from Math Training DB (especially for math queries)
        if is_math_query:
            maths_client = self._get_maths_mongo_connection()
            if maths_client:
                try:
                    db = maths_client[self.config.MATHS_TRAINING_DB]
                    collection = db[self.config.MATHS_TRAIN_COLLECTION]
                    
                    for doc in collection.find().limit(100):
                        score = self._semantic_similarity(query, doc)
                        if score > 0.1:
                            thinking_text = doc.get('thinking', 'Mathematical reasoning and step-by-step solution')
                            contexts.append({
                                'score': score * 1.2,
                                'input': doc.get('input', ''),
                                'output': doc.get('output', ''),
                                'thinking': thinking_text,
                                'mood': 'analytical',
                                'similarity_type': 'math_semantic',
                                'source': 'math_training'
                            })
                except Exception as e:
                    print(f"❌ Error in math training context retrieval: {e}")
                finally:
                    self._close_maths_mongo_connection(maths_client)
        
        # Also search in lang_model data for creative context
        if self._tokenizer and len(contexts) < max_contexts:
            lang_contexts = self._search_lang_model_context(query, max_contexts - len(contexts))
            contexts.extend(lang_contexts)
        
        # Sort by semantic score and take top ones
        contexts.sort(key=lambda x: x['score'], reverse=True)
        top_contexts = contexts[:max_contexts]
        
        # Cache the results
        self._context_cache[cache_key] = top_contexts
        
        if top_contexts:
            best_score = top_contexts[0]['score']
            sources = set(ctx['source'] for ctx in top_contexts)
            print(f"🔍 Semantic search: {len(top_contexts)} contexts, best score: {best_score:.3f}, sources: {sources}")
        
        return top_contexts
    
    def _semantic_similarity(self, query: str, document: Dict) -> float:
        """Calculate semantic similarity between query and document"""
        if not self._tokenizer:
            return self._keyword_similarity(query, document)
        
        try:
            query_tokens = set(self._tokenizer.encode(query))
            doc_text = f"{document.get('input', '')} {document.get('output', '')} {document.get('thinking', '')}"
            doc_tokens = set(self._tokenizer.encode(doc_text))
            
            if not query_tokens or not doc_tokens:
                return 0.0
            
            intersection = len(query_tokens.intersection(doc_tokens))
            union = len(query_tokens.union(doc_tokens))
            
            if union == 0:
                return 0.0
            
            length_penalty = min(1.0, 100 / len(doc_tokens))
            return (intersection / union) * length_penalty
            
        except Exception as e:
            print(f"⚠️ Semantic similarity error: {e}")
            return self._keyword_similarity(query, document)
    
    def _keyword_similarity(self, query: str, document: Dict) -> float:
        """Fallback keyword-based similarity"""
        query_words = set(query.lower().split())
        doc_text = f"{document.get('input', '')} {document.get('output', '')}".lower()
        doc_words = set(doc_text.split())
        
        if not query_words:
            return 0.0
        
        common_words = query_words.intersection(doc_words)
        return len(common_words) / len(query_words)
    
    def _search_lang_model_context(self, query: str, max_contexts: int) -> List[Dict]:
        """Search for relevant context in lang_model data"""
        if not self._tokenizer:
            return []
        
        try:
            contexts = []
            query_tokens = set(self._tokenizer.encode(query.lower()))
            
            for text in self.load_lang_model_data(max_lines=100):
                text_tokens = set(self._tokenizer.encode(text.lower()))
                intersection = len(query_tokens.intersection(text_tokens))
                
                if intersection > 0:
                    score = intersection / len(query_tokens) if query_tokens else 0
                    contexts.append({
                        'score': score * 0.5,
                        'input': 'Creative writing request',
                        'output': text[:200] + '...' if len(text) > 200 else text,
                        'thinking': 'Providing creative writing context',
                        'mood': 'creative',
                        'similarity_type': 'lang_model_semantic',
                        'source': 'lang_model'
                    })
                
                if len(contexts) >= max_contexts * 2:
                    break
            
            return contexts[:max_contexts]
            
        except Exception as e:
            print(f"⚠️ Lang model context search error: {e}")
            return []
    
    def analyze_data_size(self) -> Tuple[int, int]:
        """Analyze data for auto-scaling with mixed sources"""
        sample_batch = []
        mixed_stream = self.get_training_pairs_streaming(batch_size=1000)
        
        for batch in mixed_stream:
            sample_batch.extend(batch)
            if len(sample_batch) >= 5000:
                break
        
        data_size = len(sample_batch)
        
        if self._tokenizer:
            total_tokens = 0
            unique_tokens = set()
            
            for pair in tqdm(sample_batch, desc="🔍 Counting tokens"):
                input_tokens = self._tokenizer.encode(pair['input'])
                output_tokens = self._tokenizer.encode(pair['output'])
                thinking_tokens = self._tokenizer.encode(pair.get('thinking', ''))
                
                total_tokens += len(input_tokens) + len(output_tokens) + len(thinking_tokens)
                unique_tokens.update(input_tokens + output_tokens + thinking_tokens)
            
            unique_token_count = len(unique_tokens)
            avg_tokens_per_example = total_tokens / len(sample_batch) if sample_batch else 0
            
            print(f"📊 Mixed data analysis: {data_size} examples, {unique_token_count} unique tokens")
            print(f"📏 Average tokens per example: {avg_tokens_per_example:.1f}")
            
        else:
            all_text = " ".join([f"{p['input']} {p['output']} {p.get('thinking', '')}" 
                               for p in sample_batch])
            unique_tokens = len(set(all_text.split()))
            unique_token_count = unique_tokens
        
        return data_size, unique_token_count
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get detailed data statistics with mixed source info"""
        sample_batch = []
        mixed_stream = self.get_training_pairs_streaming(batch_size=1000)
        
        for batch in mixed_stream:
            sample_batch.extend(batch)
            if len(sample_batch) >= 3000:
                break
        
        if not sample_batch:
            return {"error": "No data available"}
        
        # Calculate source distribution
        source_count = defaultdict(int)
        for item in sample_batch:
            source_count[item.get('source', 'unknown')] += 1
        
        moods = [p['mood'] for p in sample_batch]
        thinking_present = sum(1 for p in sample_batch if p.get('thinking') and len(p['thinking']) > 10)
        context_used = sum(1 for p in sample_batch if p.get('context_used'))
        
        # Calculate average lengths
        if self._tokenizer:
            total_input_tokens = sum(len(self._tokenizer.encode(p['input'])) for p in sample_batch)
            total_output_tokens = sum(len(self._tokenizer.encode(p['output'])) for p in sample_batch)
            total_thinking_tokens = sum(len(self._tokenizer.encode(p.get('thinking', ''))) for p in sample_batch)
            
            avg_input_len = total_input_tokens / len(sample_batch)
            avg_output_len = total_output_tokens / len(sample_batch)
            avg_thinking_len = total_thinking_tokens / len(sample_batch)
        else:
            avg_input_len = sum(len(p['input'].split()) for p in sample_batch) / len(sample_batch)
            avg_output_len = sum(len(p['output'].split()) for p in sample_batch) / len(sample_batch)
            avg_thinking_len = sum(len(p.get('thinking', '').split()) for p in sample_batch) / len(sample_batch)
        
        return {
            "total_examples": len(sample_batch),
            "data_sources": dict(source_count),
            "mood_distribution": {mood: moods.count(mood) for mood in set(moods)},
            "thinking_present": thinking_present,
            "context_used": context_used,
            "avg_input_length": round(avg_input_len, 1),
            "avg_output_length": round(avg_output_len, 1),
            "avg_thinking_length": round(avg_thinking_len, 1),
            "tokenizer_used": self._tokenizer is not None,
            "persistent_mongo": self.persistent_mongo,
            "smart_context": True,
            "math_training_available": self.config.MATHS_TRAINING_URL is not None
        }
    
    def close_persistent_connections(self):
        """Close persistent connections"""
        if self._mongo_client:
            self._close_mongo_connection(self._mongo_client)
            self._mongo_client = None
            print("🔒 Persistent MongoDB connection closed")
        
        if self._maths_mongo_client:
            self._close_maths_mongo_connection(self._maths_mongo_client)
            self._maths_mongo_client = None
            print("🔒 Persistent Math Training MongoDB connection closed")
    
    def __del__(self):
        """Destructor to ensure connections are closed"""
        self.close_persistent_connections()
        