import os
import math
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LANG_MODEL_PATH = os.path.join(BASE_DIR, "data", "lang_model.txt")
    MONGODB_URL = os.getenv("MONGODB_URL")
    DATABASE_NAME = "elf_owl_db" 
    COLLECTION_NAME = "dataset"
    
    # New Math Training Data Source
    MATHS_TRAINING_URL = os.getenv("MATHS_TRAINING")
    MATHS_TRAINING_DB = "dataset"
    MATHS_TRAIN_COLLECTION = "NumCrunch"
    
    # Auto-scaling parameters (will be set automatically)
    D_MODEL = 256  # Set a reasonable default
    N_LAYERS = 6  
    N_HEADS = 8
    D_FF = 1024
    VOCAB_SIZE = 50000
    
    # Fixed parameters
    DROPOUT = 0.1
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    EPOCHS = 15
    MAX_SEQUENCE_LENGTH = 512
    
    # Generation parameters
    MAX_GENERATION_LENGTH = 512
    MIN_GENERATION_LENGTH = 1
    TOP_K = 50
    TOP_P = 0.85
    TEMPERATURE = 0.7
    REPETITION_PENALTY = 1.5
    MAX_MONGO_EXAMPLES = 100000
    MAX_MATHS_EXAMPLES = 100000  
    
    # Training monitoring
    SAMPLE_PROMPTS = ["Hello", "How are you?", "Who created you?", "What is mass of H?", "Solve 2+2", "Calculate derivative of x^2"]
    
    # Mood system - Added math-specific moods
    MOODS = [
    # Core moods
    "playful",      # 😄 Fun, humorous, lighthearted
    "curious",      # 🤔 Inquisitive, asking questions
    "analytical",   # 🔍 Logical, detailed, methodical  
    "empathetic",   # 💝 Caring, understanding, supportive
    "creative",     # 🎨 Imaginative, artistic, story-telling
    "sarcastic",    # 😏 Witty, ironic, teasing (lightly)
    "wise",         # 🧠 Knowledgeable, philosophical
    "formal",       # 📋 Professional, structured
    
    # Additional moods
    "enthusiastic", # 🎉 Excited, energetic, positive
    "calm",         # 🍃 Peaceful, soothing, reassuring
    "mysterious",   # 🔮 Cryptic, intriguing, enigmatic
    "dramatic",     # 🎭 Theatrical, exaggerated, suspenseful
    "friendly",     # 👋 Warm, welcoming, approachable
    "professional", # 💼 Business-like, expert, precise
    "romantic",     # 💖 Passionate, affectionate, poetic
    "adventurous",  # 🗺️ Bold, exploratory, daring
    "humorous",     # 😂 Funny, joke-telling, entertaining
    "serious",      # 🎯 Focused, no-nonsense, direct
    "whimsical",    # ✨ Fanciful, dreamy, magical
    "scientific",   # 🔬 Fact-based, evidence-driven
    "poetic",       # 📜 Lyrical, metaphorical, beautiful
    "confident",    # 💪 Assertive, self-assured, bold
    "humble",       # 🙏 Modest, self-effacing, gracious
    "rebellious",   # ⚡ Challenging norms, unconventional
    
    # Math-specific moods
    "precise",      # 🎯 Exact, accurate, step-by-step
    "logical",      # 🧩 Structured, reasoning-based
    "educational"   # 📚 Teaching, explanatory
    ]
    
    MOOD_DESCRIPTIONS = {
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
    "rebellious": "Challenging norms and unconventional ideas",
    "precise": "Exact, accurate, and step-by-step explanations",
    "logical": "Structured, reasoning-based problem solving",
    "educational": "Teaching-oriented and explanatory approach"
    }
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    DEBUG = True
    
    # Model paths
    MODEL_SAVE_PATH = "models/elf_owl_model"
    TOKENIZER_SAVE_PATH = "models/tokenizer"
    
    @classmethod
    def auto_scale(cls, data_size: int, unique_tokens: int):
        """Auto-scale model based on data size and vocabulary"""
        # Scale d_model based on data (32 to 512)
        if data_size < 1000:
            cls.D_MODEL = 32
        elif data_size < 10000:
            cls.D_MODEL = 128  
        elif data_size < 50000:
            cls.D_MODEL = 256
        else:
            cls.D_MODEL = 512
        
        # Scale layers
        cls.N_LAYERS = max(2, min(8, cls.D_MODEL // 64))
        cls.N_HEADS = max(2, min(8, cls.D_MODEL // 32))
        cls.D_FF = cls.D_MODEL * 4
        
        # Scale vocab size
        cls.VOCAB_SIZE = min(50000, max(1000, unique_tokens + 1000))
        
        print(f"🔧 Auto-scaled: d_model={cls.D_MODEL}, layers={cls.N_LAYERS}, vocab={cls.VOCAB_SIZE}")
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('_') and not callable(v)}
    