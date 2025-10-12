import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import re
import random
import time
from datetime import datetime

from data_loader import DataLoader
from tokenizer import ElfOwlTokenizer
from model import ElfOwlModel
import config

class ElfOwlInference:
    def __init__(self, model_path: str = None, tokenizer_path: str = None, 
                 persistent_mongo: bool = True, use_smart_context: bool = True):
        self.config = config.Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_smart_context = use_smart_context
        self._default_mood = None  # Add this line
        
        # Initialize components
        self.data_loader = DataLoader(persistent_mongo=persistent_mongo)
        self.tokenizer = ElfOwlTokenizer()
        
        # Load tokenizer
        if tokenizer_path is None:
            tokenizer_path = self.config.TOKENIZER_SAVE_PATH
        self.tokenizer.load(tokenizer_path)
        
        # Set tokenizer for smart features
        self.data_loader.set_tokenizer(self.tokenizer)
        
        # Get special tokens
        self.special_tokens = self.tokenizer.get_special_tokens()
        
        # Load model
        self.model = self.load_model(model_path)
        if self.model:
            self.model.eval()
        
        # Response cache for similar queries
        self.response_cache = {}
        self.cache_hits = 0
        self.total_queries = 0
        
        print("🚀 Elf Owl AI Inference Engine Ready!")
        print(f"🔧 Device: {self.device}")
        print(f"🎯 Free Generation: {self.config.MIN_GENERATION_LENGTH} to {self.config.MAX_GENERATION_LENGTH} tokens")
        print(f"🧠 Smart Context: {use_smart_context}")
        print(f"🔗 Persistent MongoDB: {persistent_mongo}")
    
    def load_model(self, model_path: str = None) -> Optional[ElfOwlModel]:
        """Load trained model - FIXED version"""
        if model_path is None:
            model_path = f"{self.config.MODEL_SAVE_PATH}.pt"
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Use saved config or current config
            saved_config = checkpoint.get('config_dict', {})
            vocab_size = checkpoint.get('vocab_size', self.tokenizer.vocab_size)
            
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
    
    def get_smart_context(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Get context using semantic search from both MongoDB and lang_model"""
        if not self.use_smart_context:
            return self.get_basic_context(query, conversation_history)
        
        try:
            # Get semantically relevant contexts
            relevant_contexts = self.data_loader.get_smart_factual_context(query)
            
            if not relevant_contexts:
                return ""
            
            # Build context string from multiple sources
            context_parts = []
            for ctx in relevant_contexts:
                # Include similarity score for debugging (optional)
                score_info = f"[score:{ctx['score']:.2f}]" if ctx.get('score') else ""
                
                if ctx.get('similarity_type') == 'lang_model_semantic':
                    context_parts.append(f"Creative: {ctx['output']} {score_info}")
                else:
                    context_parts.append(f"Factual: {ctx['input']} -> {ctx['output']} {score_info}")
            
            context = " | ".join(context_parts)
            
            best_score = relevant_contexts[0].get('score', 0) if relevant_contexts else 0
            print(f"🧠 Smart context: {len(relevant_contexts)} sources, best: {best_score:.3f}")
            
            return context
            
        except Exception as e:
            print(f"⚠️ Smart context error: {e}, falling back to basic context")
            return self.get_basic_context(query, conversation_history)
    
    def get_basic_context(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Fallback context retrieval using basic keyword matching"""
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
        
        # Add basic factual context
        try:
            basic_contexts = self.data_loader.get_factual_context_for_inference(query)
            for ctx in basic_contexts[:2]:
                context_parts.append(f"Factual: {ctx['input']} -> {ctx['output']}")
        except Exception as e:
            print(f"⚠️ Basic context error: {e}")
        
        context = " | ".join(context_parts) if context_parts else ""
        
        if context:
            print(f"🔍 Basic context: {len(context_parts)} sources")
        
        return context
    
    def _get_cached_response(self, query: str, context: str) -> Optional[Dict[str, Any]]:
        """Check cache for similar queries"""
        self.total_queries += 1
        
        # Simple cache key based on query and context
        cache_key = hash(f"{query.lower().strip()}:{context.lower().strip()}")
        
        # Check cache (valid for 5 minutes)
        if cache_key in self.response_cache:
            cached_time, response = self.response_cache[cache_key]
            if time.time() - cached_time < 300:  # 5 minutes
                self.cache_hits += 1
                print(f"💾 Cache hit: {self.cache_hits}/{self.total_queries} "
                      f"({self.cache_hits/self.total_queries*100:.1f}%)")
                return response
        
        return None
    
    def _cache_response(self, query: str, context: str, response: Dict[str, Any]):
        """Cache response for similar future queries"""
        cache_key = hash(f"{query.lower().strip()}:{context.lower().strip()}")
        self.response_cache[cache_key] = (time.time(), response)
        
        # Limit cache size
        if len(self.response_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self.response_cache.keys(), 
                               key=lambda k: self.response_cache[k][0])[:100]
            for key in oldest_keys:
                del self.response_cache[key]
    
    def generate_response(self, prompt: str, conversation_history: List[Dict] = None, 
                         max_length: int = None, temperature: float = None, 
                         top_k: int = None, top_p: float = None,
                         use_cache: bool = True, 
                         forced_mood: str = None) -> Dict[str, Any]:  # FIXED: forced_mood parameter defined
        """Generate response with optional mood selection"""
        if not self.model:
            return {
                "response": "Model not loaded. Please check if the model file exists.",
                "thinking": "System error: Model not available",
                "mood": "neutral",
                "context_used": False,
                "tokens_generated": 0,
                "error": "model_not_loaded"
            }
        
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
        
        # Get context (smart or basic)
        context = self.get_smart_context(prompt, conversation_history)
        
        # Check cache
        if use_cache:
            cached_response = self._get_cached_response(prompt, context)
            if cached_response:
                cached_response['cached'] = True
                cached_response['response_time'] = time.time() - start_time
                return cached_response
        
        # MOOD SELECTION: Use forced mood or auto-detect
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
    
        # Build structured prompt
        prompt_parts = []
        
        # Add context if available
        if context:
            prompt_parts.append(f"Context: {context}")
        
        # Add current input
        prompt_parts.append(f"Input: {prompt}")
        
        # Build the full prompt for thinking generation
        thinking_prompt = " [SEP] ".join(prompt_parts) + " [SEP] Thinking:"
        
        # Generate thinking step
        thinking = self._generate_text(thinking_prompt, max_length=150, temperature=0.7)
        thinking = self._clean_response(thinking)
        
        # Build final prompt with thinking
        response_prompt = thinking_prompt + f" {thinking} [SEP] Mood: {mood} [SEP] Output:"
        
        # Generate final response
        response = self._generate_text(response_prompt, max_length=max_length, 
                                     temperature=temperature, top_k=top_k, top_p=top_p)
        response = self._clean_response(response)
        
        # Build result
        result = {
            "response": response,
            "thinking": thinking,
            "mood": mood,
            "mood_source": mood_source,
            "context_used": bool(context),
            "tokens_generated": len(self.tokenizer.encode(response)),
            "response_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }
        
        # Cache the response
        if use_cache and result['tokens_generated'] > 0:
            self._cache_response(prompt, context, result)
        
        print(f"✅ Generated {result['tokens_generated']} tokens in {result['response_time']:.2f}s")
        
        return result
    
    def _generate_text(self, prompt: str, max_length: int = 100, 
                      temperature: float = 0.7, top_k: int = None, top_p: float = None) -> str:
        """Generate text from prompt with enhanced controls"""
        if top_k is None:
            top_k = self.config.TOP_K
        if top_p is None:
            top_p = self.config.TOP_P
        
        input_ids = self.tokenizer.encode(prompt)
        
        # Handle very long prompts by truncating from the beginning
        max_input_length = self.config.MAX_SEQUENCE_LENGTH - max_length - 10
        if len(input_ids) > max_input_length:
            # Keep the most relevant part (assume end is more important)
            input_ids = input_ids[-max_input_length:]
            print(f"⚠️ Truncated input from {len(input_ids) + max_input_length} to {len(input_ids)} tokens")
        
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                input_tensor,
                max_length=len(input_ids) + max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=self.config.REPETITION_PENALTY,
                eos_token_id=self.special_tokens.get("[EOS]")
            )
        
        # Extract response
        response_tokens = generated[0].cpu().tolist()[len(input_ids):]
        if self.special_tokens["[EOS]"] in response_tokens:
            eos_idx = response_tokens.index(self.special_tokens["[EOS]"])
            response_tokens = response_tokens[:eos_idx]
        
        return self.tokenizer.decode(response_tokens)
    
    def _select_mood(self, prompt: str) -> str:
        """Select appropriate mood based on query with enhanced detection"""
        prompt_lower = prompt.lower()
        
        # Mood detection with priority
        mood_rules = [
            (['joke', 'funny', 'laugh', 'haha', 'lol'], "playful"),
            (['sad', 'depressed', 'unhappy', 'cry', 'hurt'], "empathetic"),
            (['angry', 'mad', 'hate', 'annoyed', 'frustrated'], "calm"),
            (['love', 'romantic', 'heart', 'crush'], "romantic"),
            (['story', 'creative', 'imagine', 'write', 'poem'], "creative"),
            (['why', 'how', 'explain', 'what is', 'tell me about'], "analytical"),
            (['help', 'problem', 'issue', 'trouble', 'support'], "helpful"),
            (['?', 'who', 'when', 'where', 'which'], "curious"),
            (['thank', 'thanks', 'appreciate'], "grateful"),
            (['sorry', 'apologize', 'forgive'], "forgiving"),
            (['stupid', 'dumb', 'idiot', 'useless'], "patient"),
        ]
        
        for keywords, mood in mood_rules:
            if any(keyword in prompt_lower for keyword in keywords):
                return mood
        
        # Default based on query characteristics
        if len(prompt.split()) > 20:
            return "wise"
        elif len(prompt.split()) < 3:
            return "friendly"
        else:
            return random.choice(self.config.MOODS)
    
    def _clean_response(self, response: str) -> str:
        """Clean and format response with enhanced cleaning"""
        if not response:
            return ""
        
        # Remove special tokens
        for token in self.special_tokens:
            response = response.replace(token, '')
        
        # Remove any remaining [SEP] markers and other artifacts
        response = re.sub(r'\[SEP\]', '', response)
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Fix common issues
        response = re.sub(r'\s+([.,!?;])', r'\1', response)  # Remove spaces before punctuation
        response = re.sub(r'(\w)\s+\.', r'\1.', response)    # Fix spaced periods
        
        # Ensure proper sentence casing
        if response:
            response = response[0].upper() + response[1:]
        
        # Ensure proper ending
        if response and response[-1] not in ['.', '!', '?', '"', "'", ':']:
            # Add appropriate ending based on content
            if any(marker in response.lower() for marker in ['question', 'what', 'why', 'how']):
                response += '?'
            elif any(marker in response.lower() for marker in ['great', 'wonderful', 'excellent']):
                response += '!'
            else:
                response += '.'
        
        # Fix common grammatical issues
        response = re.sub(r'\bi\b', 'I', response)
        response = re.sub(r'\bim\b', "I'm", response)
        response = re.sub(r'\bive\b', "I've", response)
        
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
        return {
            "playful": "Fun, humorous, and lighthearted responses",
            "curious": "Asking questions and exploring ideas", 
            "analytical": "Logical, detailed, and methodical thinking",
            "empathetic": "Caring, understanding, and supportive tone",
            "creative": "Imaginative, artistic, and story-telling",
            "sarcastic": "Witty, ironic, and lightly teasing",
            "wise": "Knowledgeable and philosophical perspective",
            "formal": "Professional and structured communication",
            "enthusiastic": "Excited, energetic, and positive vibes",
            "calm": "Peaceful, soothing, and reassuring tone",
            "mysterious": "Cryptic, intriguing, and enigmatic style",
            "dramatic": "Theatrical, exaggerated, and suspenseful",
            "friendly": "Warm, welcoming, and approachable manner",
            "professional": "Business-like and expert communication", 
            "romantic": "Passionate, affectionate, and poetic language",
            "adventurous": "Bold, exploratory, and daring attitude",
            "humorous": "Funny, joke-telling, and entertaining",
            "serious": "Focused, no-nonsense, and direct approach",
            "whimsical": "Fanciful, dreamy, and magical thinking",
            "scientific": "Fact-based and evidence-driven responses",
            "poetic": "Lyrical, metaphorical, and beautiful language",
            "confident": "Assertive, self-assured, and bold statements",
            "humble": "Modest, self-effacing, and gracious tone",
            "rebellious": "Challenging norms and unconventional ideas"
        }
    
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
            "smart_context_enabled": self.use_smart_context,
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "default_mood": self._default_mood
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        self.cache_hits = 0
        self.total_queries = 0
        print("🧹 Response cache cleared")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'data_loader'):
            self.data_loader.close_persistent_connections()
            print("🔒 Inference resources cleaned up")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()