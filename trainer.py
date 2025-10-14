import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import json
from datetime import datetime
import time
from typing import Iterator

from data_loader import DataLoader as ElfOwlDataLoader
from tokenizer import ElfOwlTokenizer
from model import ElfOwlModel
import config

class ElfOwlDataset(Dataset):
    def __init__(self, training_pairs, tokenizer, max_length=512):
        self.pairs = training_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.special_tokens = tokenizer.get_special_tokens()
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # FORMAT: Input -> Thinking -> Mood -> Output
        text_parts = [
            f"Input: {pair['input']}",
            f"Thinking: {pair.get('thinking', 'Analyzing the query...')}",
            f"Mood: {pair['mood']}",
            f"Output: {pair['output']}"
        ]
        
        text = " [SEP] ".join(text_parts) + " [EOS]"
        tokens = self.tokenizer.encode(text)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input and target (shifted by one)
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        # Pad if too short
        if len(input_ids) < self.max_length:
            padding = [self.special_tokens["[PAD]"]] * (self.max_length - len(input_ids))
            input_ids = input_ids + padding
            target_ids = target_ids + padding
        
        return torch.tensor(input_ids), torch.tensor(target_ids)

class StreamingDataset(Iterator):
    """Dataset for streaming mixed data batches"""
    def __init__(self, data_loader, tokenizer, max_length=512, batch_size=1000):
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.special_tokens = tokenizer.get_special_tokens()
        self._stream = None
        self._current_batch = []
        self._batch_index = 0
        
    def __iter__(self):
        self._stream = self.data_loader.get_training_pairs_streaming(batch_size=self.batch_size)
        self._batch_index = 0
        return self
    
    def __next__(self):
        if self._stream is None:
            self._stream = self.data_loader.get_training_pairs_streaming(batch_size=self.batch_size)
        
        try:
            batch_pairs = next(self._stream)
            self._batch_index += 1
            
            # Convert batch to tensors
            input_batch = []
            target_batch = []
            
            for pair in batch_pairs:
                # Format the text
                text_parts = [
                    f"Input: {pair['input']}",
                    f"Thinking: {pair.get('thinking', 'Analyzing the query...')}",
                    f"Mood: {pair['mood']}",
                    f"Output: {pair['output']}"
                ]
                
                text = " [SEP] ".join(text_parts) + " [EOS]"
                tokens = self.tokenizer.encode(text)
                
                # Truncate if too long
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                
                # Create input and target
                input_ids = tokens[:-1]
                target_ids = tokens[1:]
                
                # Pad if too short
                if len(input_ids) < self.max_length:
                    padding = [self.special_tokens["[PAD]"]] * (self.max_length - len(input_ids))
                    input_ids = input_ids + padding
                    target_ids = target_ids + padding
                
                input_batch.append(torch.tensor(input_ids))
                target_batch.append(torch.tensor(target_ids))
            
            return torch.stack(input_batch), torch.stack(target_batch)
            
        except StopIteration:
            self._stream = None
            raise StopIteration

class ElfOwlTrainer:
    def __init__(self, persistent_mongo: bool = True, use_streaming: bool = False):
        self.config = config.Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_streaming = use_streaming
        print(f"🚀 Using device: {self.device}")
        print(f"📦 Training mode: {'Streaming' if use_streaming else 'Batch'}")
        print(f"🔗 Persistent MongoDB: {persistent_mongo}")
        
        # Initialize components
        self.data_loader = ElfOwlDataLoader(persistent_mongo=persistent_mongo)
        self.tokenizer = ElfOwlTokenizer()
        self.model = None
        self.optimizer = None
        
        # Training stats
        self.training_stats = {
            'start_time': None,
            'total_batches': 0,
            'total_examples': 0,
            'best_loss': float('inf')
        }
    
    def prepare_training(self):
        """Prepare everything for training"""
        print("🔄 Preparing training...")
        
        # Load data for analysis
        print("📚 Loading training data for analysis...")
        training_pairs = self.data_loader.get_training_pairs(max_examples=5000)
        data_size = len(training_pairs)
        
        # Calculate unique tokens
        unique_words = set()
        for pair in training_pairs:
            unique_words.update(pair['input'].split())
            unique_words.update(pair['output'].split())
            if pair.get('thinking'):
                unique_words.update(pair['thinking'].split())
        
        unique_token_count = len(unique_words)
        print(f"📊 Data analysis: {data_size} examples, {unique_token_count} unique words")
        
        # Auto-scale
        self.config.auto_scale(data_size, unique_token_count)
        
        # Train tokenizer
        print("🔄 Training tokenizer...")
        self.tokenizer.train_tokenizer(self.data_loader)
        
        # Set tokenizer for data_loader
        self.data_loader.set_tokenizer(self.tokenizer)
        
        # Create dataset
        print("📚 Creating dataset...")
        dataset = ElfOwlDataset(training_pairs, self.tokenizer, self.config.MAX_SEQUENCE_LENGTH)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0
        )
        
        # Initialize model
        self.model = ElfOwlModel(self.tokenizer.vocab_size).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.EPOCHS,
            eta_min=self.config.LEARNING_RATE * 0.1
        )
        
        print("✅ Training preparation complete!")
        
        return {
            "data_size": data_size,
            "unique_tokens": unique_token_count,
            "training_examples": len(training_pairs),
            "model_parameters": total_params
        }
    
    def create_dataloader(self):
        """Create appropriate dataloader based on streaming preference"""
        if self.use_streaming:
            print("🎯 Using streaming dataloader with mixed sources...")
            streaming_dataset = StreamingDataset(
                self.data_loader, 
                self.tokenizer, 
                self.config.MAX_SEQUENCE_LENGTH,
                batch_size=self.config.BATCH_SIZE
            )
            return streaming_dataset
        else:
            print("🎯 Using batch dataloader...")
            training_pairs = self.data_loader.get_training_pairs()
            dataset = ElfOwlDataset(training_pairs, self.tokenizer, self.config.MAX_SEQUENCE_LENGTH)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.BATCH_SIZE, 
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            return dataloader
    
    def sample_generation(self, epoch, global_step=0):
        """Generate sample responses to monitor progress"""
        self.model.eval()
        special_tokens = self.tokenizer.get_special_tokens()
        
        print(f"\n🎯 Epoch {epoch} (Step {global_step}) Sample Generations:")
        print("-" * 50)
        
        for prompt in self.config.SAMPLE_PROMPTS:
            # Test full sequence generation
            input_text = f"Input: {prompt} [SEP] Thinking:"
            input_ids = self.tokenizer.encode(input_text)
            input_tensor = torch.tensor([input_ids]).to(self.device)
            
            with torch.no_grad():
                generated = self.model.generate(
                    input_tensor,
                    max_length=min(200, self.config.MAX_GENERATION_LENGTH),
                    temperature=0.8,
                    top_k=30,
                    eos_token_id=special_tokens["[EOS]"],
                    repetition_penalty=self.config.REPETITION_PENALTY
                )
            
            response_tokens = generated[0].cpu().tolist()[len(input_ids):]
            if special_tokens["[EOS]"] in response_tokens:
                eos_idx = response_tokens.index(special_tokens["[EOS]"])
                response_tokens = response_tokens[:eos_idx]
            
            full_response = self.tokenizer.decode(response_tokens)
            
            # Try to parse thinking and output
            thinking = ""
            output = full_response
            
            if " [SEP] Mood: " in full_response:
                parts = full_response.split(" [SEP] Mood: ")
                if len(parts) > 1:
                    thinking = parts[0].replace("Thinking: ", "").strip()
                    mood_output = parts[1]
                    if " [SEP] Output: " in mood_output:
                        output_parts = mood_output.split(" [SEP] Output: ")
                        if len(output_parts) > 1:
                            output = output_parts[1].strip()
            
            print(f"Q: {prompt}")
            print(f"Thinking: {thinking}")
            print(f"Response: {output}")
            print()
        
        self.model.train()
    
    def train_epoch_batch(self, dataloader, epoch):
        """Train for one epoch using batch mode"""
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device, non_blocking=True)
            target_ids = target_ids.to(self.device, non_blocking=True)
            
            # Forward pass
            logits, loss = self.model(input_ids, targets=target_ids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            self.training_stats['total_batches'] += 1
            self.training_stats['total_examples'] += input_ids.size(0)
            
            # Update progress bar
            if batch_idx % 10 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                    'lr': f'{current_lr:.2e}'
                })
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def train_epoch_streaming(self, streaming_dataset, epoch):
        """Train for one epoch using streaming mode"""
        total_loss = 0
        batch_count = 0
        
        # Create progress bar for streaming
        progress_bar = tqdm(desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [Streaming]")
        
        for batch_idx, (input_ids, target_ids) in enumerate(streaming_dataset):
            if input_ids is None:
                break
                
            input_ids = input_ids.to(self.device, non_blocking=True)
            target_ids = target_ids.to(self.device, non_blocking=True)
            
            # Forward pass
            logits, loss = self.model(input_ids, targets=target_ids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            self.training_stats['total_batches'] += 1
            self.training_stats['total_examples'] += input_ids.size(0)
            
            # Update progress bar
            if batch_count % 10 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/batch_count:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'batches': batch_count
                })
                progress_bar.update(10)
        
        progress_bar.close()
        avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
        return avg_loss
    
    def train(self):
        """Main training loop with streaming support"""
        print("🔥 Starting Elf Owl AI Training!")
        self.training_stats['start_time'] = datetime.now()
        
        # Prepare training
        stats = self.prepare_training()
        
        # Create dataloader (streaming or batch)
        dataloader = self.create_dataloader()
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.config.EPOCHS):
            epoch_start_time = time.time()
            
            # Train one epoch
            if self.use_streaming:
                avg_loss = self.train_epoch_streaming(dataloader, epoch)
            else:
                avg_loss = self.train_epoch_batch(dataloader, epoch)
            
            epoch_time = time.time() - epoch_start_time
            
            # Step scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            print(f"📈 Epoch {epoch+1} completed.")
            print(f"   Average Loss: {avg_loss:.4f}")
            print(f"   Learning Rate: {current_lr:.2e}")
            print(f"   Epoch Time: {epoch_time:.2f}s")
            print(f"   Total Examples: {self.training_stats['total_examples']:,}")
            
            # Sample generations to monitor progress
            if (epoch + 1) % 2 == 0:  # Every 2 epochs
                self.sample_generation(epoch + 1, self.training_stats['total_batches'])
            
            # Save checkpoint if loss improved
            if avg_loss < self.training_stats['best_loss']:
                self.training_stats['best_loss'] = avg_loss
                self.save_checkpoint(epoch + 1, avg_loss)
                print(f"💾 New best loss: {avg_loss:.4f}")
            
            # Save periodic checkpoint every 5 epochs
            elif (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, avg_loss, best=False)
        
        # Save final model
        self.save_model()
        self.tokenizer.save(self.config.TOKENIZER_SAVE_PATH)
        
        # Cleanup
        self.cleanup()
        
        # Print final stats
        total_time = (datetime.now() - self.training_stats['start_time']).total_seconds()
        print("\n🎉 Training completed!")
        print(f"⏰ Total training time: {total_time/60:.1f} minutes")
        print(f"📊 Best loss achieved: {self.training_stats['best_loss']:.4f}")
        print(f"📦 Total examples processed: {self.training_stats['total_examples']:,}")
        print(f"🔄 Total training batches: {self.training_stats['total_batches']:,}")
    
    def save_checkpoint(self, epoch, loss, best=True):
        """Save training checkpoint"""
        os.makedirs("checkpoints", exist_ok=True)

        prefix = "best" if best else f"epoch_{epoch}"
        checkpoint_path = f"checkpoints/elf_owl_{prefix}.pt"

        # Make a serializable copy of training_stats
        serializable_stats = dict(self.training_stats)
        if isinstance(serializable_stats.get('start_time'), datetime):
            serializable_stats['start_time'] = serializable_stats['start_time'].isoformat()

        # Create config dictionary
        config_dict = {
            'D_MODEL': self.config.D_MODEL,
            'N_LAYERS': self.config.N_LAYERS,
            'N_HEADS': self.config.N_HEADS,
            'D_FF': self.config.D_FF,
            'VOCAB_SIZE': self.config.VOCAB_SIZE,
            'DROPOUT': self.config.DROPOUT,
            'BATCH_SIZE': self.config.BATCH_SIZE,
            'LEARNING_RATE': self.config.LEARNING_RATE,
            'EPOCHS': self.config.EPOCHS,
            'MAX_SEQUENCE_LENGTH': self.config.MAX_SEQUENCE_LENGTH,
            'MAX_GENERATION_LENGTH': self.config.MAX_GENERATION_LENGTH,
            'MIN_GENERATION_LENGTH': self.config.MIN_GENERATION_LENGTH,
            'TOP_K': self.config.TOP_K,
            'TOP_P': self.config.TOP_P,
            'TEMPERATURE': self.config.TEMPERATURE,
            'REPETITION_PENALTY': self.config.REPETITION_PENALTY
        }

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config_dict': config_dict,
            'vocab_size': self.tokenizer.vocab_size,
            'training_stats': serializable_stats,
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_model(self):
        """Save final model"""
        os.makedirs("models", exist_ok=True)
        serializable_stats = dict(self.training_stats)
        if isinstance(serializable_stats.get('start_time'), datetime):
            serializable_stats['start_time'] = serializable_stats['start_time'].isoformat()

        # Create config dictionary
        config_dict = {
            'D_MODEL': self.config.D_MODEL,
            'N_LAYERS': self.config.N_LAYERS,
            'N_HEADS': self.config.N_HEADS,
            'D_FF': self.config.D_FF,
            'VOCAB_SIZE': self.config.VOCAB_SIZE,
            'DROPOUT': self.config.DROPOUT,
            'MAX_SEQUENCE_LENGTH': self.config.MAX_SEQUENCE_LENGTH,
            'MAX_GENERATION_LENGTH': self.config.MAX_GENERATION_LENGTH
        }

        model_data = {
            'model_state_dict': self.model.state_dict(),
            'config_dict': config_dict,
            'vocab_size': self.tokenizer.vocab_size,
            'training_mode': 'streaming' if self.use_streaming else 'batch',
            'training_stats': serializable_stats,
            'timestamp': datetime.now().isoformat()
        }

        model_path = f"{self.config.MODEL_SAVE_PATH}.pt"
        torch.save(model_data, model_path)
        print(f"💾 Model saved to {model_path}")
    
    def cleanup(self):
        """Cleanup resources after training"""
        if hasattr(self, 'data_loader'):
            self.data_loader.close_persistent_connections()
            print("🔒 Resources cleaned up")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Elf Owl AI')
    parser.add_argument('--streaming', action='store_true', 
                       help='Use streaming mode for large datasets')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create config instance and update from command line
    conf = config.Config()
    if args.epochs:
        conf.EPOCHS = args.epochs
    if args.batch_size:
        conf.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        conf.LEARNING_RATE = args.learning_rate

    use_streaming = args.streaming

    print("🦉 Elf Owl AI Trainer")
    print(f"🔧 Epochs: {conf.EPOCHS}")
    print(f"📦 Batch Size: {conf.BATCH_SIZE}")
    print(f"📚 Learning Rate: {conf.LEARNING_RATE}")
    print(f"🎯 Streaming Mode: {use_streaming}")

    # Pass config instance to trainer
    trainer = ElfOwlTrainer(persistent_mongo=True, use_streaming=use_streaming)
    trainer.config = conf

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.cleanup()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        trainer.cleanup()
        raise

if __name__ == "__main__":
    main()
    