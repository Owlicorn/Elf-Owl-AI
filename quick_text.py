# quick_test.py
from inference import ElfOwlInference
import time

def quick_test():
    print("🧪 Quick Model Test...")
    
    try:
        elf_owl = ElfOwlInference()
        
        # Test math capabilities
        test_questions = [
            "Calculate 15 + 27",
            "Solve 3x + 5 = 20", 
            "What is 25% of 80?",
            "Hello, how are you?",
            "Who created you?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Q: {question}")
            start_time = time.time()
            response = elf_owl.generate_response(question)
            end_time = time.time()
            
            print(f"💭 A: {response['response']}")
            print(f"🤔 Thinking: {response.get('thinking', 'N/A')}")
            print(f"🎭 Mood: {response['mood']}")
            print(f"⏱️ Time: {end_time - start_time:.2f}s")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    quick_test()