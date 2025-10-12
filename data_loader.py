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
        self._context_cache = {}  # Cache for semantic search results
        self._last_cache_clear = time.time()
        
        # For mixing streams
        self._lang_model_buffer = deque()
        self._mongo_buffer = deque()
        self._min_buffer_size = 100
        
        # Initialize connections if persistent mode
        if persistent_mongo:
            self._get_mongo_connection()
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for proper token counting and semantic search"""
        self._tokenizer = tokenizer
    
    def _get_mongo_connection(self) -> Optional[MongoClient]:
        """Get MongoDB connection with persistence option"""
        if self._mongo_client and self.persistent_mongo:
            try:
                # Test if connection is still alive
                self._mongo_client.admin.command('ping')
                return self._mongo_client
            except:
                # Connection died, reset it
                self._mongo_client = None
        
        if not self.config.MONGODB_URL:
            return None
        
        try:
            client = MongoClient(
                self.config.MONGODB_URL,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=30000,
                maxPoolSize=20,  # Increased for persistent connections
                minPoolSize=5 if self.persistent_mongo else 1,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=10000
            )
            
            # Test connection
            client.admin.command('ping')
            
            if self.persistent_mongo:
                self._mongo_client = client
                print("🔗 Persistent MongoDB connection established")
            
            return client
            
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            return None
    
    def _close_mongo_connection(self, client):
        """Close MongoDB connection gracefully (unless persistent)"""
        if client and (client is not self._mongo_client or not self.persistent_mongo):
            try:
                client.close()
            except Exception as e:
                print(f"⚠️ Error closing MongoDB connection: {e}")
    
    def _semantic_similarity(self, query: str, document: Dict) -> float:
        """Calculate semantic similarity between query and document"""
        if not self._tokenizer:
            # Fallback to keyword matching
            return self._keyword_similarity(query, document)
        
        try:
            # Simple semantic similarity using token overlap and weighting
            query_tokens = set(self._tokenizer.encode(query))
            doc_text = f"{document.get('input', '')} {document.get('output', '')} {document.get('thinking', '')}"
            doc_tokens = set(self._tokenizer.encode(doc_text))
            
            if not query_tokens or not doc_tokens:
                return 0.0
            
            # Jaccard similarity
            intersection = len(query_tokens.intersection(doc_tokens))
            union = len(query_tokens.union(doc_tokens))
            
            if union == 0:
                return 0.0
            
            # Weight by document length (prefer shorter, more relevant docs)
            length_penalty = min(1.0, 100 / len(doc_tokens))  # Prefer documents under 100 tokens
            
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
    
    def _fill_buffers(self, target_size: int = 500):
        """Fill both lang_model and MongoDB buffers for mixed streaming. Returns True if any new data was added, False if both sources are exhausted."""
        lang_model_exhausted = False
        mongo_exhausted = False
        lang_added = 0
        mongo_added = 0

        # Fill lang_model buffer
        if len(self._lang_model_buffer) < target_size:
            lang_stream = self.load_lang_model_data(max_lines=target_size * 2)
            for text in lang_stream:
                self._lang_model_buffer.append({
                    "input": "Write a creative piece of text",
                    "thinking": "I should generate creative text with good grammar and flow",
                    "output": text,
                    "mood": random.choice(["creative", "playful", "wise", "descriptive"]),
                    "context_used": [],
                    "source": "lang_model"
                })
                lang_added += 1
                if len(self._lang_model_buffer) >= target_size * 2:
                    break
            if lang_added == 0:
                lang_model_exhausted = True
        else:
            lang_model_exhausted = False

        # Fill MongoDB buffer
        if len(self._mongo_buffer) < target_size:
            mongo_stream = self.load_mongodb_conversations(limit=target_size * 2)
            for conv in mongo_stream:
                input_text = conv.get('input') or conv.get('user_input') or conv.get('question', '')
                output_text = conv.get('output') or conv.get('assistant_response') or conv.get('answer', '')

                if input_text and output_text and len(input_text.strip()) > 2 and len(output_text.strip()) > 2:
                    self._mongo_buffer.append({
                        "input": input_text.strip(),
                        "thinking": conv.get('thinking', 'Analyzing the user query and formulating a helpful response').strip(),
                        "output": output_text.strip(),
                        "mood": conv.get('mood', random.choice(self.config.MOODS)),
                        "context_used": conv.get('context_used', []),
                        "source": "mongodb"
                    })
                    mongo_added += 1
                    if len(self._mongo_buffer) >= target_size * 2:
                        break
            if mongo_added == 0:
                mongo_exhausted = True
        else:
            mongo_exhausted = False

        # If both sources could not add any new data, signal exhaustion
        return not (lang_model_exhausted and mongo_exhausted)
    
    def load_lang_model_data(self, max_lines: int = None) -> Iterator[str]:
        """Load raw text data for language modeling with streaming support"""
        if not os.path.exists(self.config.LANG_MODEL_PATH):
            print(f"❌ lang_model.txt not found at {self.config.LANG_MODEL_PATH}")
            return
        
        line_count = 0
        try:
            with open(self.config.LANG_MODEL_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
                        line_count += 1
                        
                        if line_count % 1000 == 0:
                            time.sleep(0.001)
                        
                        if max_lines and line_count >= max_lines:
                            break
            
            print(f"✅ Loaded {line_count} lines from lang_model.txt")
            
        except Exception as e:
            print(f"❌ Error reading lang_model.txt: {e}")
    
    def load_mongodb_conversations(self, limit: int = None) -> Iterator[Dict]:
        """Load structured conversations from MongoDB with streaming support"""
        client = self._get_mongo_connection()
        if not client:
            return
        
        try:
            db = client[self.config.DATABASE_NAME]
            collection = db[self.config.COLLECTION_NAME]
            
            cursor = collection.find({})
            if limit:
                cursor = cursor.limit(limit)
            
            count = 0
            for doc in cursor:
                yield doc
                count += 1
                
                if count % 1000 == 0:
                    time.sleep(0.001)
            
            print(f"✅ Streamed {count} conversations from MongoDB")
            
        except Exception as e:
            print(f"❌ MongoDB streaming error: {e}")
        finally:
            self._close_mongo_connection(client)
    
    def get_training_pairs(self, max_examples: int = None) -> List[Dict[str, Any]]:
        """Convert all data to training format with mixed sources"""
        training_data = []
        total_pairs_processed = 0
        
        # Use mixed streaming for better convergence
        mixed_stream = self.get_training_pairs_streaming(batch_size=1000)
        
        for batch in mixed_stream:
            for example in batch:
                training_data.append(example)
                total_pairs_processed += 1
                
                if max_examples and total_pairs_processed >= max_examples:
                    break
            
            if max_examples and total_pairs_processed >= max_examples:
                break
        
        # Final shuffle
        random.shuffle(training_data)
        
        print(f"📚 Total training pairs: {len(training_data)}")
        
        # Show mixing ratio
        if training_data:
            lang_count = sum(1 for d in training_data if d.get('source') == 'lang_model')
            mongo_count = len(training_data) - lang_count
            print(f"📊 Mixed ratio: {lang_count} lang_model + {mongo_count} MongoDB examples")
        
        return training_data
    
    def get_training_pairs(self, max_examples: int = 5000) -> List[Dict[str, Any]]:
        """Convert all data to training format - FIXED for finite data"""
        print("📚 Loading training data (non-streaming mode)...")
        training_data = []
        
        # Load lang_model data (finite)
        lang_texts = list(self.load_lang_model_data(max_lines=2000))
        print(f"📄 Loaded {len(lang_texts)} lines from lang_model.txt")
        
        for text in lang_texts:
            training_data.append({
                "input": "Write a creative piece of text",
                "thinking": "I should generate creative text with good grammar and flow",
                "output": text,
                "mood": random.choice(["creative", "playful", "wise", "descriptive"]),
                "context_used": []
            })
        
        # Load MongoDB conversations (finite)  
        mongo_conversations = list(self.load_mongodb_conversations(limit=5000))
        print(f"🗄️ Loaded {len(mongo_conversations)} conversations from MongoDB")
        
        for conv in mongo_conversations:
            input_text = conv.get('input') or conv.get('user_input') or conv.get('question', '')
            output_text = conv.get('output') or conv.get('assistant_response') or conv.get('answer', '')
            
            if input_text and output_text and len(input_text.strip()) > 2 and len(output_text.strip()) > 2:
                training_data.append({
                    "input": input_text.strip(),
                    "thinking": conv.get('thinking', 'Analyzing the user query and formulating a helpful response').strip(),
                    "output": output_text.strip(),
                    "mood": conv.get('mood', random.choice(self.config.MOODS)),
                    "context_used": conv.get('context_used', [])
                })
        
        # Final shuffle and limit
        random.shuffle(training_data)
        
        if max_examples and len(training_data) > max_examples:
            training_data = training_data[:max_examples]
        
        print(f"📚 Total training pairs: {len(training_data)}")
        
        # Show composition
        lang_count = len([d for d in training_data if "creative piece" in d["input"]])
        mongo_count = len(training_data) - lang_count
        print(f"📊 Composition: {lang_count} lang_model + {mongo_count} MongoDB examples")
        
        return training_data
    
    def get_smart_factual_context(self, query: str, max_contexts: int = 3) -> List[Dict]:
        """SMARTER CONTEXT RETRIEVAL using semantic search"""
        # Clear cache every hour to prevent staleness
        if time.time() - self._last_cache_clear > 3600:
            self._context_cache.clear()
            self._last_cache_clear = time.time()
        
        # Check cache first
        cache_key = hash(query.lower().strip())
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        client = self._get_mongo_connection()
        if not client:
            return []
        
        try:
            db = client[self.config.DATABASE_NAME]
            collection = db[self.config.COLLECTION_NAME]
            
            # Get candidate documents
            candidates = []
            for doc in collection.find().limit(200):  # Increased for better semantic search
                score = self._semantic_similarity(query, doc)
                if score > 0.05:  # Minimum similarity threshold
                    candidates.append({
                        'score': score,
                        'input': doc.get('input', ''),
                        'output': doc.get('output', ''),
                        'thinking': doc.get('thinking', ''),
                        'mood': doc.get('mood', 'neutral'),
                        'similarity_type': 'semantic' if self._tokenizer else 'keyword'
                    })
            
            # Also search in lang_model data for creative context
            if self._tokenizer and len(candidates) < max_contexts:
                lang_contexts = self._search_lang_model_context(query, max_contexts - len(candidates))
                candidates.extend(lang_contexts)
            
            # Sort by semantic score and take top ones
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top_candidates = candidates[:max_contexts]
            
            # Cache the results
            self._context_cache[cache_key] = top_candidates
            
            if top_candidates:
                best_score = top_candidates[0]['score']
                print(f"🔍 Semantic search: {len(top_candidates)} contexts, best score: {best_score:.3f}")
            
            return top_candidates
            
        except Exception as e:
            print(f"❌ Error in smart context retrieval: {e}")
            return []
        finally:
            self._close_mongo_connection(client)
    
    def _search_lang_model_context(self, query: str, max_contexts: int) -> List[Dict]:
        """Search for relevant context in lang_model data"""
        if not self._tokenizer:
            return []
        
        try:
            # Sample some lang_model data for context
            contexts = []
            query_tokens = set(self._tokenizer.encode(query.lower()))
            
            for text in self.load_lang_model_data(max_lines=100):
                text_tokens = set(self._tokenizer.encode(text.lower()))
                intersection = len(query_tokens.intersection(text_tokens))
                
                if intersection > 0:
                    score = intersection / len(query_tokens) if query_tokens else 0
                    contexts.append({
                        'score': score * 0.5,  # Lower weight for lang_model contexts
                        'input': 'Creative writing request',
                        'output': text[:200] + '...' if len(text) > 200 else text,  # Truncate long texts
                        'thinking': 'Providing creative writing context',
                        'mood': 'creative',
                        'similarity_type': 'lang_model_semantic'
                    })
                
                if len(contexts) >= max_contexts * 2:  # Get extra for filtering
                    break
            
            return contexts[:max_contexts]
            
        except Exception as e:
            print(f"⚠️ Lang model context search error: {e}")
            return []
    
    def analyze_data_size(self) -> Tuple[int, int]:
        """Analyze data for auto-scaling with mixed sources"""
        # Sample from mixed stream for accurate analysis
        sample_batch = []
        mixed_stream = self.get_training_pairs_streaming(batch_size=1000)
        
        for batch in mixed_stream:
            sample_batch.extend(batch)
            if len(sample_batch) >= 5000:  # Sample size
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
            "smart_context": True
        }
    
    def close_persistent_connections(self):
        """Close persistent connections (call this at the end of long training)"""
        if self._mongo_client:
            self._close_mongo_connection(self._mongo_client)
            self._mongo_client = None
            print("🔒 Persistent MongoDB connection closed")
    
    def __del__(self):
        """Destructor to ensure connections are closed"""
        self.close_persistent_connections()