import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from typing import List, Dict
import config

class ElfOwlTokenizer:
    def __init__(self):
        self.config = config.Config()
        self.tokenizer = None
        self.vocab_size = None
        
        # Special tokens
        self.special_tokens = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "[EOS]", "[THINK]", "[MOOD]"
        ]
    
    def train_tokenizer(self, data_loader):
        """Train BPE tokenizer on combined data with special tokens"""
        print("🔄 Training BPE tokenizer...")
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Get training data for tokenizer
        training_pairs = data_loader.get_training_pairs(max_examples=5000)
        
        # Prepare text for tokenizer training
        texts = []
        for pair in training_pairs:
            texts.append(pair['input'])
            texts.append(pair['output'])
            if pair.get('thinking'):
                texts.append(pair['thinking'])
        
        # **FIX: Handle None VOCAB_SIZE**
        if self.config.VOCAB_SIZE is None:
            # Calculate reasonable vocab size from data
            unique_words = set()
            for text in texts:
                unique_words.update(text.split())
            vocab_size = min(50000, len(unique_words) + len(self.special_tokens))
            print(f"⚠️ VOCAB_SIZE was None, using calculated size: {vocab_size}")
        else:
            vocab_size = self.config.VOCAB_SIZE
        
        print(f"📝 Using vocabulary size: {vocab_size}")
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2
        )
        
        # Train tokenizer
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        print(f"✅ Tokenizer trained with {self.vocab_size} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not trained")
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not trained")
        return self.tokenizer.decode(token_ids)
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings"""
        return {token: self.tokenizer.token_to_id(token) for token in self.special_tokens}
    
    def save(self, path: str):
        """Save tokenizer"""
        if self.tokenizer:
            self.tokenizer.save(path)
            print(f"💾 Tokenizer saved to {path}")
    
    def load(self, path: str):
        """Load tokenizer"""
        if os.path.exists(path):
            self.tokenizer = Tokenizer.from_file(path)
            self.vocab_size = self.tokenizer.get_vocab_size()
            print(f"📂 Tokenizer loaded from {path}")
        else:
            print(f"❌ Tokenizer file not found at {path}")