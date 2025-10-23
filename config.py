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
    D_MODEL = 512
    N_LAYERS = 8
    N_HEADS = 8
    D_FF = 2048
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
    
    # Data loading limits - ENFORCE THESE STRICTLY
    MAX_MONGO_EXAMPLES = 40000
    MAX_MATHS_EXAMPLES = 30000
    MAX_LANG_MODEL_LINES = 50000
    
    # Training monitoring
    SAMPLE_PROMPTS = ["Hello", "How are you?", "Who created you?", "What is mass of H?", "Solve 2+2", "Calculate derivative of x^2"]
    
    # Mood system
    MOODS = [
        "playful", "curious", "analytical", "empathetic", "creative", 
        "sarcastic", "wise", "formal", "enthusiastic", "calm", 
        "mysterious", "dramatic", "friendly", "professional", "romantic",
        "adventurous", "humorous", "serious", "whimsical", "scientific",
        "poetic", "confident", "humble", "rebellious", "precise", 
        "logical", "educational"
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
    def auto_scale(cls, data_size: int, unique_tokens: int, total_estimated_data: int = None):
        """Auto-scale model based on FULL data size, not just sample"""
        # Use the larger of actual sample or estimated total
        effective_size = total_estimated_data if total_estimated_data and total_estimated_data > data_size else data_size
        
        print(f"📊 Auto-scaling for {effective_size:,} total examples (sample: {data_size:,}), {unique_tokens:,} unique tokens")
        
        # AGGRESSIVE SCALING FOR LARGE DATASETS
        if effective_size >= 80000:  # Your actual data size ~100K
            cls.D_MODEL = 1024
            cls.N_LAYERS = 12
            cls.N_HEADS = 16
            cls.D_FF = 4096
            cls.VOCAB_SIZE = min(100000, unique_tokens + 10000)
            print("🚀 LARGE MODEL: 100K+ examples detected")
            
        elif effective_size >= 50000:
            cls.D_MODEL = 768
            cls.N_LAYERS = 10
            cls.N_HEADS = 12
            cls.D_FF = 3072
            cls.VOCAB_SIZE = min(75000, unique_tokens + 5000)
            print("📈 MEDIUM-LARGE MODEL: 50K+ examples detected")
            
        elif effective_size >= 20000:
            cls.D_MODEL = 512
            cls.N_LAYERS = 8
            cls.N_HEADS = 8
            cls.D_FF = 2048
            cls.VOCAB_SIZE = min(50000, unique_tokens + 3000)
            print("📊 MEDIUM MODEL: 20K+ examples detected")
            
        elif effective_size >= 10000:
            cls.D_MODEL = 384
            cls.N_LAYERS = 6
            cls.N_HEADS = 6
            cls.D_FF = 1536
            cls.VOCAB_SIZE = min(40000, unique_tokens + 2000)
            print("📚 SMALL-MEDIUM MODEL: 10K+ examples detected")
            
        else:
            cls.D_MODEL = 256
            cls.N_LAYERS = 4
            cls.N_HEADS = 4
            cls.D_FF = 1024
            cls.VOCAB_SIZE = min(30000, unique_tokens + 1000)
            print("🔬 SMALL MODEL: <10K examples")
        
        # Calculate estimated parameters
        embedding_params = cls.VOCAB_SIZE * cls.D_MODEL
        attention_params = cls.N_LAYERS * (4 * cls.D_MODEL * cls.D_MODEL)
        ff_params = cls.N_LAYERS * (2 * cls.D_MODEL * cls.D_FF)
        total_params = embedding_params + attention_params + ff_params
        estimated_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per parameter
        
        print(f"🎯 Final Model: d_model={cls.D_MODEL}, layers={cls.N_LAYERS}, heads={cls.N_HEADS}")
        print(f"📈 Model size: {total_params:,} params, ~{estimated_mb:.1f}MB")
        print(f"📚 Vocabulary: {cls.VOCAB_SIZE}")
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('_') and not callable(v)}