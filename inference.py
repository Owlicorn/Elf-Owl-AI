import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import re
import random
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import json
import urllib.parse
import os

from data_loader import DataLoader
from tokenizer import ElfOwlTokenizer
from model import ElfOwlModel
import config

class ElfOwlInference:
    def __init__(self, model_path: str = None, tokenizer_path: str = None, 
                 persistent_mongo: bool = True, use_smart_context: bool = True,
                 enable_web_search: bool = True):
        self.config = config.Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_smart_context = use_smart_context
        self.enable_web_search = enable_web_search
        self._default_mood = None
        
        # Initialize components
        self.data_loader = DataLoader(persistent_mongo=persistent_mongo)
        self.tokenizer = ElfOwlTokenizer()
        
        # Load tokenizer - DON'T TRAIN, JUST LOAD OR FAIL GRACEFULLY
        if tokenizer_path is None:
            tokenizer_path = self.config.TOKENIZER_SAVE_PATH
        
        if os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)
            print(f"✅ Tokenizer loaded from {tokenizer_path}")
        else:
            print(f"❌ Tokenizer not found at {tokenizer_path}")
            print("💡 Please train the model first: python trainer.py")
            # Set a flag to indicate tokenizer is not ready
            self._tokenizer_ready = False
            return
        
        self._tokenizer_ready = True
        
        # Set tokenizer for smart features
        self.data_loader.set_tokenizer(self.tokenizer)
        
        # Get special tokens
        self.special_tokens = self.tokenizer.get_special_tokens()
        
        # Load model
        self.model = self.load_model(model_path)
        if self.model:
            self.model.eval()
            print("✅ Model loaded successfully!")
        else:
            print("❌ Model not loaded - please train first")
        
        # Response cache for similar queries
        self.response_cache = {}
        self.cache_hits = 0
        self.total_queries = 0
        
        # Web search cache to avoid repeated searches
        self.web_search_cache = {}
        
        # Fallback knowledge base for common questions
        self.fallback_knowledge = {
            "current pm of india": "Narendra Modi is the current Prime Minister of India. He has been serving since 2014.",
            "population of india": "As of 2024, India's population is approximately 1.44 billion people, making it the most populous country in the world.",
            "capital of india": "New Delhi is the capital of India.",
            "current president of india": "Droupadi Murmu is the current President of India, serving since 2022.",
            "current pm of usa": "Joe Biden is the current President of the United States.",
            "current president of usa": "Joe Biden is the current President of the United States.",
            "capital of usa": "Washington D.C. is the capital of the United States.",
            "population of china": "China's population is approximately 1.425 billion people as of 2024.",
            "current pm of uk": "Rishi Sunak is the current Prime Minister of the United Kingdom.",
            "current pm of canada": "Justin Trudeau is the current Prime Minister of Canada.",
            "solve 2+2": "2 + 2 = 4",
            "calculate 5*5": "5 × 5 = 25",
            "what is 10/2": "10 ÷ 2 = 5",
            "square root of 16": "√16 = 4",
            "pi value": "π ≈ 3.14159",
            "who created you": "I was created by Puneet Kumar Mishra on 12-10-2025.",
            "what is your name": "I am Elf Owl AI, a virtual owl AI created by Puneet Kumar Mishra.",
            "hello": "Hello! I'm Elf Owl AI, how can I help you today? 🦉",
            "how are you": "I'm doing great! Ready to help with any questions or calculations you have! 🦉"
        }
        
        print("🚀 Elf Owl AI Inference Engine Ready!")
        print(f"🔧 Device: {self.device}")
        print(f"🎯 Free Generation: {self.config.MIN_GENERATION_LENGTH} to {self.config.MAX_GENERATION_LENGTH} tokens")
        print(f"🧠 Smart Context: {use_smart_context}")
        print(f"🔍 Web Search: {enable_web_search}")
        print(f"🔗 Persistent MongoDB: {persistent_mongo}")
        print(f"🔢 Math Training Available: {self.config.MATHS_TRAINING_URL is not None}")
        print(f"📚 Fallback Knowledge: {len(self.fallback_knowledge)} entries")
        print(f"✅ Tokenizer Ready: {self._tokenizer_ready}")
        print(f"✅ Model Ready: {self.model is not None}")
    
    def load_model(self, model_path: str = None) -> Optional[ElfOwlModel]:
        """Load trained model"""
        if model_path is None:
            model_path = f"{self.config.MODEL_SAVE_PATH}.pt"
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at {model_path}")
            print("💡 Please train the model first using: python trainer.py")
            return None
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Use saved config or current config
            saved_config = checkpoint.get('config_dict', {})
            vocab_size = checkpoint.get('vocab_size', self.tokenizer.vocab_size if self._tokenizer_ready else 50000)
            
            # Update config with saved values if available
            if saved_config:
                for key, value in saved_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            print(f"🔧 Loading model: d_model={self.config.D_MODEL}, vocab={vocab_size}")
            
            model = ElfOwlModel(vocab_size).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model loaded successfully!")
            
            # Print training info if available
            if 'training_stats' in checkpoint:
                stats = checkpoint['training_stats']
                print(f"📊 Training info: {stats.get('total_examples', 0):,} examples, "
                    f"best loss: {stats.get('best_loss', 0):.4f}")
            
            return model
        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    def _clean_user_query(self, query: str) -> str:
        """Clean and normalize user query"""
        if not query:
            return ""
        
        # Remove extra spaces and normalize
        query = ' '.join(query.split())
        
        # Remove repeated phrases (handle spamming)
        words = query.split()
        if len(words) > 10:  # Likely spam/repeated
            # Take first few unique words
            unique_words = []
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)
                if len(unique_words) >= 8:  # Reasonable query length
                    break
            query = ' '.join(unique_words)
        
        return query.lower().strip()
    
    def _get_fallback_answer(self, query: str) -> Optional[str]:
        """Get answer from fallback knowledge base"""
        clean_query = self._clean_user_query(query)
        
        # Exact match
        if clean_query in self.fallback_knowledge:
            return self.fallback_knowledge[clean_query]
        
        # Partial match
        for key, value in self.fallback_knowledge.items():
            if key in clean_query or clean_query in key:
                return value
        
        return None
    
    def _is_gibberish_thinking(self, thinking: str) -> bool:
        """Detect if the thinking process is gibberish/untrained output"""
        if not thinking or len(thinking.strip()) < 10:
            return True
            
        thinking_lower = thinking.lower()
        
        # Gibberish indicators
        gibberish_indicators = [
            "ouuuuu", "oooooh", "🦉", "💖", "✨", "his", "circuits", "digital",
            "vast", "fledgling", "owlet", "hoot", "wisdom", "tweet", "chirp"
        ]
        
        # Count gibberish indicators
        gibberish_count = sum(1 for indicator in gibberish_indicators if indicator in thinking_lower)
        
        # If more than 2 gibberish indicators, likely untrained output
        if gibberish_count >= 2:
            return True
        
        # Check for very short or repetitive content
        words = thinking.split()
        if len(words) < 5:
            return True
            
        # Check for reasonable sentence structure
        if thinking_lower.count('.') == 0 and thinking_lower.count('?') == 0 and thinking_lower.count('!') == 0:
            return True
            
        return False
    
    def _needs_web_search(self, query: str) -> bool:
        """Determine if web search is needed based on query"""
        query_lower = self._clean_user_query(query)
        
        # Check if we have a fallback answer first
        if self._get_fallback_answer(query):
            print("📚 Using fallback knowledge, skipping web search")
            return False
        
        # Keywords that indicate need for current/recent information
        search_keywords = [
            'current', 'recent', 'latest', 'today', 'yesterday', 'this week', 'this month',
            'new', 'update', 'breaking', 'news', '2024', '2025', 'now', 'live',
            'what happened', 'when did', 'who is', 'where is', 'how to',
            'population of', 'capital of', 'president of', 'prime minister of',
            'weather', 'stock', 'crypto', 'sports', 'score'
        ]
        
        # Check if query requires current information
        if any(keyword in query_lower for keyword in search_keywords):
            return True
        
        # Specific factual questions
        specific_questions = [
            'who is', 'what is', 'when did', 'where is', 'how many',
            'how much', 'how old', 'how to'
        ]
        
        if any(query_lower.startswith(phrase) for phrase in specific_questions):
            return True
            
        return False
    
    def _search_web_simple(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Simple web search with fallback to knowledge base"""
        if not self.enable_web_search:
            return []
        
        # Check cache first
        cache_key = hash(query.lower())
        if cache_key in self.web_search_cache:
            print("🔍 Using cached web search results")
            return self.web_search_cache[cache_key]
        
        print(f"🌐 Searching web for: {query}")
        
        # First, check if we have a fallback answer
        fallback_answer = self._get_fallback_answer(query)
        if fallback_answer:
            print("✅ Using fallback knowledge")
            return [{
                'source': 'Fallback Knowledge',
                'content': fallback_answer,
                'url': ''
            }]
        
        # If no fallback, try web APIs
        results = []
        
        try:
            # Method 1: DuckDuckGo Instant Answer API
            print("🦆 Trying DuckDuckGo API...")
            ddg_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(ddg_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract from Abstract
                if data.get('Abstract') and data['Abstract']:
                    results.append({
                        'source': 'DuckDuckGo',
                        'content': data['Abstract'],
                        'url': data.get('AbstractURL', '')
                    })
                    print("✅ Found DuckDuckGo abstract")
            
            # If still no results, return empty and rely on fallback later
            if not results:
                print("❌ No web results found, will use fallback")
            
            # Cache the results (even if empty)
            results = results[:max_results]
            self.web_search_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            print(f"❌ Web search error: {e}")
            return []
    
    def _extract_relevant_info(self, web_results: List[Dict], query: str) -> str:
        """Extract and summarize relevant information from web results"""
        if not web_results:
            # Try fallback knowledge as last resort
            fallback = self._get_fallback_answer(query)
            if fallback:
                return f"Fallback Knowledge: {fallback}"
            return ""
        
        relevant_info = []
        
        for result in web_results:
            content = result.get('content', '')
            source = result.get('source', 'Unknown')
            
            if content:
                # Truncate long content
                truncated_content = content[:200] + "..." if len(content) > 200 else content
                relevant_info.append(f"{source}: {truncated_content}")
        
        # Combine all relevant information
        if relevant_info:
            combined_info = " | ".join(relevant_info)
            print(f"📄 Extracted {len(relevant_info)} relevant snippets")
            return combined_info
        else:
            return ""
    
    def get_smart_context(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Get context using semantic search from all sources"""
        if not self.use_smart_context:
            return self.get_basic_context(query, conversation_history)
        
        try:
            # Get semantically relevant contexts from ALL sources
            relevant_contexts = self.data_loader.get_smart_factual_context(query)
            
            context_parts = []
            
            # Add database contexts
            for ctx in relevant_contexts:
                source_info = f"[{ctx.get('source', 'unknown')}]"
                
                if ctx.get('similarity_type') == 'lang_model_semantic':
                    context_parts.append(f"Creative {source_info}: {ctx['output']}")
                elif ctx.get('source') == 'math_training':
                    context_parts.append(f"Math {source_info}: {ctx['input']} -> {ctx['output']}")
                else:
                    context_parts.append(f"Factual {source_info}: {ctx['input']} -> {ctx['output']}")
            
            # Combine database contexts
            database_context = " | ".join(context_parts) if context_parts else ""
            
            if relevant_contexts:
                best_score = relevant_contexts[0].get('score', 0)
                sources = set(ctx.get('source', 'unknown') for ctx in relevant_contexts)
                print(f"🧠 Smart context: {len(relevant_contexts)} sources from {sources}, best: {best_score:.3f}")
            
            return database_context
            
        except Exception as e:
            print(f"⚠️ Smart context error: {e}, falling back to basic context")
            return self.get_basic_context(query, conversation_history)
    
    def get_basic_context(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Fallback context retrieval"""
        context_parts = []
        
        # Add conversation history context
        if conversation_history and len(conversation_history) > 0:
            recent = conversation_history[-2:]
            for turn in recent:
                if isinstance(turn, dict) and 'user' in turn and 'assistant' in turn:
                    user_msg = str(turn.get('user', '')).strip()
                    assistant_msg = str(turn.get('assistant', '')).strip()
                    if user_msg and assistant_msg:
                        context_parts.append(f"Previous: {user_msg} -> {assistant_msg}")
        
        context = " | ".join(context_parts) if context_parts else ""
        
        if context:
            print(f"🔍 Basic context: {len(context_parts)} sources")
        
        return context
    
    def _get_cached_response(self, query: str, context: str) -> Optional[Dict[str, Any]]:
        """Check cache for similar queries"""
        clean_query = self._clean_user_query(query)
        cache_key = hash(f"{clean_query}:{context}")
        
        if cache_key in self.response_cache:
            cached_time, response = self.response_cache[cache_key]
            if time.time() - cached_time < 300:  # 5 minutes
                self.cache_hits += 1
                self.total_queries += 1
                print(f"💾 Cache hit: {self.cache_hits}/{self.total_queries} ({self.cache_hits/self.total_queries*100:.1f}%)")
                return response
        
        self.total_queries += 1
        return None
    
    def _cache_response(self, query: str, context: str, response: Dict[str, Any]):
        """Cache response for similar future queries"""
        clean_query = self._clean_user_query(query)
        cache_key = hash(f"{clean_query}:{context}")
        self.response_cache[cache_key] = (time.time(), response)
        
        if len(self.response_cache) > 1000:
            oldest_keys = sorted(self.response_cache.keys(), 
                               key=lambda k: self.response_cache[k][0])[:100]
            for key in oldest_keys:
                del self.response_cache[key]
    
    def _generate_safe_response(self, query: str, mood: str) -> Dict[str, Any]:
        """Generate a safe response when model produces gibberish"""
        clean_query = self._clean_user_query(query)
        
        # Try to get factual answer first
        factual_answer = self._get_fallback_answer(query)
        if factual_answer:
            thinking = f"I recall that {factual_answer}"
            response = f"Based on available knowledge: {factual_answer}"
        else:
            # Generic friendly response
            thinking = "I'm still learning about this topic, but I want to be helpful"
            response = f"I'm sorry, I don't have specific information about '{clean_query}' right now. I'm constantly learning and improving! 🦉"
        
        return {
            "response": response,
            "thinking": thinking,
            "mood": mood,
            "mood_source": "fallback",
            "context_used": False,
            "web_search_used": False,
            "tokens_generated": len(response.split()),
            "response_time": 0.1,
            "timestamp": datetime.now().isoformat(),
            "cached": False,
            "fallback_used": True
        }
    
    def generate_response(self, prompt: str, conversation_history: List[Dict] = None, 
                         max_length: int = None, temperature: float = None, 
                         top_k: int = None, top_p: float = None,
                         use_cache: bool = True, 
                         forced_mood: str = None) -> Dict[str, Any]:
        """Generate response with comprehensive fallback system"""
        # Check if components are ready
        if not self._tokenizer_ready:
            return self._generate_safe_response(prompt, "neutral")
        
        if not self.model:
            return self._generate_safe_response(prompt, "neutral")
        
        # Clean the prompt first
        original_prompt = prompt
        prompt = self._clean_user_query(original_prompt)
        
        if not prompt:
            return self._generate_safe_response("empty query", "neutral")
        
        # Use config defaults if not provided
        if max_length is None:
            max_length = self.config.MAX_GENERATION_LENGTH
        if temperature is None:
            temperature = self.config.TEMPERATURE
        if top_k is None:
            top_k = self.config.TOP_K
        if top_p is None:
            top_p = self.config.TOP_P
        
        start_time = time.time()
        
        # Get context from ALL sources
        context = self.get_smart_context(prompt, conversation_history)
        
        # Check cache
        if use_cache:
            cached_response = self._get_cached_response(prompt, context)
            if cached_response:
                cached_response['response_time'] = time.time() - start_time
                return cached_response
        
        # MOOD SELECTION
        if forced_mood and forced_mood in self.config.MOODS:
            mood = forced_mood
            mood_source = "user_selected"
            print(f"🎭 Using user-selected mood: {mood}")
        elif self._default_mood and self._default_mood in self.config.MOODS:
            mood = self._default_mood
            mood_source = "default_setting"
            print(f"🎭 Using default mood: {mood}")
        else:
            mood = self._select_mood(prompt)
            mood_source = "auto_detected"
            print(f"🎭 Auto-detected mood: {mood}")

        # STEP 1: CHECK IF WEB SEARCH IS NEEDED
        web_context = ""
        should_search = self._needs_web_search(prompt)
        
        if self.enable_web_search and should_search:
            print("🔍 Web search triggered based on query")
            web_results = self._search_web_simple(prompt)
            web_context = self._extract_relevant_info(web_results, prompt)
        
        # STEP 2: GENERATE THINKING WITH WEB CONTEXT IF AVAILABLE
        thinking_prompt_parts = []
        if context:
            thinking_prompt_parts.append(f"Context: {context}")
        if web_context:
            thinking_prompt_parts.append(f"Web Context: {web_context}")
        thinking_prompt_parts.append(f"Input: {prompt}")
        
        thinking_prompt = " [SEP] ".join(thinking_prompt_parts) + " [SEP] Thinking:"
        
        # Generate thinking
        thinking = self._generate_text(
            thinking_prompt, 
            max_length=100, 
            temperature=0.7,
            stop_tokens=["[SEP]", "[EOS]"]
        )
        thinking = self._clean_response(thinking)
        print(f"🤔 Thinking: {thinking}")

        # STEP 3: CHECK IF THINKING IS GIBBERISH
        if self._is_gibberish_thinking(thinking):
            print("⚠️ Thinking is gibberish, using fallback response")
            safe_response = self._generate_safe_response(prompt, mood)
            safe_response['response_time'] = time.time() - start_time
            # Don't cache gibberish responses
            return safe_response

        # STEP 4: GENERATE FINAL OUTPUT
        output_prompt_parts = thinking_prompt_parts.copy()
        output_prompt_parts.extend([
            f"Thinking: {thinking}",
            f"Mood: {mood}",
            "Output:"
        ])
        
        output_prompt = " [SEP] ".join(output_prompt_parts)
        
        # Generate output
        response = self._generate_text(
            output_prompt, 
            max_length=max_length, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p,
            stop_tokens=["[EOS]"]
        )
        response = self._clean_response(response)
        
        # Check if response is also gibberish
        if not response or len(response.strip()) < 5:
            print("⚠️ Response is empty/gibberish, using fallback")
            safe_response = self._generate_safe_response(prompt, mood)
            safe_response['response_time'] = time.time() - start_time
            return safe_response
        
        print(f"💭 Response: {response}")

        # Build result
        result = {
            "response": response,
            "thinking": thinking,
            "mood": mood,
            "mood_source": mood_source,
            "context_used": bool(context),
            "web_search_used": bool(web_context),
            "tokens_generated": len(self.tokenizer.encode(response)),
            "response_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "cached": False,
            "fallback_used": False
        }
        
        # Only cache good responses
        if result['tokens_generated'] > 5 and not self._is_gibberish_thinking(response):
            self._cache_response(prompt, context, result)
        
        print(f"✅ Generated {result['tokens_generated']} tokens in {result['response_time']:.2f}s")
        
        return result
    
    def _generate_text(self, prompt: str, max_length: int = 100, 
                      temperature: float = 0.7, top_k: int = None, top_p: float = None,
                      stop_tokens: List[str] = None) -> str:
        """Generate text with proper stop token handling"""
        if top_k is None:
            top_k = self.config.TOP_K
        if top_p is None:
            top_p = self.config.TOP_P
        if stop_tokens is None:
            stop_tokens = ["[SEP]", "[EOS]"]
        
        input_ids = self.tokenizer.encode(prompt)
        
        max_input_length = self.config.MAX_SEQUENCE_LENGTH - max_length - 10
        if len(input_ids) > max_input_length:
            input_ids = input_ids[-max_input_length:]
        
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        stop_token_ids = []
        for token in stop_tokens:
            if token in self.special_tokens:
                stop_token_ids.append(self.special_tokens[token])
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_tensor,
                max_length=len(input_ids) + max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.special_tokens.get("[EOS]"),
                repetition_penalty=self.config.REPETITION_PENALTY
            )
        
        generated_tokens = generated_ids[0].cpu().tolist()[len(input_ids):]
        
        for stop_id in stop_token_ids:
            if stop_id in generated_tokens:
                stop_idx = generated_tokens.index(stop_id)
                generated_tokens = generated_tokens[:stop_idx]
                break
        
        return self.tokenizer.decode(generated_tokens)
    
    def _select_mood(self, prompt: str) -> str:
        """Select appropriate mood based on query"""
        prompt_lower = prompt.lower()
        
        mood_rules = [
            (['calculate', 'solve', 'math', 'equation', 'formula', 'derivative', 'integral'], "analytical"),
            (['joke', 'funny', 'laugh', 'haha', 'lol'], "playful"),
            (['sad', 'depressed', 'unhappy', 'cry', 'hurt'], "empathetic"),
            (['angry', 'mad', 'hate', 'annoyed'], "calm"),
            (['love', 'romantic', 'heart'], "romantic"),
            (['story', 'creative', 'imagine', 'write'], "creative"),
            (['why', 'how', 'explain', 'what is'], "analytical"),
            (['help', 'problem', 'issue', 'trouble'], "helpful"),
            (['?', 'who', 'when', 'where'], "curious"),
            (['thank', 'thanks'], "grateful"),
            (['hello', 'hi', 'hey'], "friendly"),
            (['serious', 'important', 'critical'], "serious"),
            (['science', 'physics', 'chemistry', 'biology'], "scientific"),
        ]
        
        for keywords, mood in mood_rules:
            if any(keyword in prompt_lower for keyword in keywords):
                return mood
        
        return random.choice(["friendly", "curious", "helpful", "analytical"])
    
    def _clean_response(self, response: str) -> str:
        """Clean and format response"""
        if not response:
            return ""
        
        # Remove special tokens
        for token in self.special_tokens:
            response = response.replace(token, '')
        
        response = re.sub(r'\[SEP\]', '', response)
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Capitalize first letter
        if response:
            response = response[0].upper() + response[1:]
        
        return response
    
    def chat(self, message: str, conversation_history: List[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Simple chat interface"""
        return self.generate_response(message, conversation_history, **kwargs)
    
    def direct_generate(self, prompt: str, **kwargs) -> str:
        """Direct generation without structured output"""
        result = self.generate_response(prompt, **kwargs)
        return result["response"]
    
    def get_available_moods(self) -> Dict[str, str]:
        """Get all available moods with descriptions"""
        return self.config.MOOD_DESCRIPTIONS
    
    def set_default_mood(self, mood: str) -> bool:
        """Set a default mood for all responses"""
        if mood in self.config.MOODS:
            self._default_mood = mood
            print(f"🎭 Default mood set to: {mood}")
            return True
        else:
            print(f"❌ Invalid mood: {mood}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        return {
            "cache_hits": self.cache_hits,
            "total_queries": self.total_queries,
            "cache_hit_rate": self.cache_hits / self.total_queries if self.total_queries > 0 else 0,
            "cache_size": len(self.response_cache),
            "web_search_cache_size": len(self.web_search_cache),
            "smart_context_enabled": self.use_smart_context,
            "web_search_enabled": self.enable_web_search,
            "model_loaded": self.model is not None,
            "tokenizer_ready": self._tokenizer_ready,
            "device": str(self.device),
            "default_mood": self._default_mood,
            "math_training_available": self.config.MATHS_TRAINING_URL is not None,
            "fallback_knowledge_size": len(self.fallback_knowledge)
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        self.web_search_cache.clear()
        self.cache_hits = 0
        self.total_queries = 0
        print("🧹 Response and web search caches cleared")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'data_loader'):
            self.data_loader.close_persistent_connections()
            print("🔒 Inference resources cleaned up")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()
        