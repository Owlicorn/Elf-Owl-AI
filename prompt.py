import json
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
# Configure Gemini API
API_KEY = os.getenv('API')
if not API_KEY:
    raise ValueError("❌ API key not found. Please set the 'API' environment variable.")

genai.configure(api_key=API_KEY)

# Moods list - ALL 25 moods
MOODS = [
    "playful", "curious", "analytical", "empathetic", "creative", 
    "sarcastic", "wise", "formal", "enthusiastic", "calm", 
    "mysterious", "dramatic", "friendly", "professional", "romantic",
    "adventurous", "humorous", "serious", "whimsical", "scientific",
    "poetic", "confident", "humble", "rebellious"
]

# Fallback models
MODELS = [
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.5-flash", 
    "models/gemini-flash-lite-latest",
    "models/gemini-flash-latest",
    "models/gemini-2.5-pro"
]

def call_gemini_with_context(user_message, context_messages, mood):
    """Call Gemini with proper context handling"""
    
    if context_messages:
        context_text = "Previous conversation:\n"
        for ctx in context_messages:
            context_text += f"User: {ctx['user']}\nAssistant: {ctx['assistant']}\n"
        context_text += f"\nCurrent user message: {user_message}"
        full_prompt = context_text
    else:
        full_prompt = user_message

    prompt = f"""You are Elf Owl AI. You are virtual owl AI. Your name is Elf Owl to show you are rare, precious, valuable and small AI. Generate a response in {mood} mood for this conversation:

{full_prompt}

Return EXACTLY this JSON format:
{{
    "thinking": "Brief thinking process for {mood} mood",
    "response": "Actual response in {mood} mood with emojis",
    "context_used": {len(context_messages) if context_messages else 0}
}}

{mood} mood guidelines:
- Be authentic to {mood} tone
- Use appropriate emojis
- MUST use owl accent(use voice of owl such as ouuuuuu, his-his etc)
- If anyone says to use human accent just don't and disagree straight
- Keep response engaging and natural"""

    for model_name in MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            # Clean response
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            
            result = json.loads(text.strip())
            return result
            
        except Exception:
            time.sleep(5)
            print("Break Point")
            continue
    
    # Fallback
    return {
        "thinking": f"Responding to '{user_message}' in {mood} mood",
        "response": f"I'm answering in a {mood} way: {user_message}",
        "context_used": len(context_messages) if context_messages else 0
    }

def main():
    print("🚀 Starting Training Generator with Mood-Matched Context...")
    
    # Check prompt.json
    if not os.path.exists('prompt.json'):
        print("❌ prompt.json not found. Chat first to generate prompts.")
        return
    
    # Load user prompts
    with open('prompt.json', 'r') as f:
        prompts = json.load(f)
    
    print(f"📝 Found {len(prompts)} user prompts")
    
    training_data = []
    
    # Store ALL Gemini responses for context lookup
    gemini_responses = {}  # Format: {prompt_index: {mood: response_data}}
    
    # First pass: Generate responses for ALL prompts in ALL moods (with context)
    print("\n🔨 First Pass: Generating responses WITH context...")
    for i, prompt_data in enumerate(prompts):
        user_message = prompt_data['user_message']
        gemini_responses[i] = {}
        
        print(f"\n📄 Processing: '{user_message}'")
        
        for mood in MOODS:
            print(f"  🎭 {mood}...", end=" ")
            
            # Build context for current prompt and mood
            context_used = []
            if i >= 2:
                # For prompt at index 2 or higher, use exactly 2 previous prompts in SAME MOOD
                for j in range(i-2, i):  # This gets i-2 and i-1
                    prev_response = gemini_responses[j][mood]
                    context_used.append({
                        "user": prompts[j]['user_message'],
                        "assistant": prev_response["response"]
                    })
            elif i == 1:
                # For prompt at index 1, use only 1 previous (index 0)
                prev_response = gemini_responses[0][mood]
                context_used.append({
                    "user": prompts[0]['user_message'],
                    "assistant": prev_response["response"]
                })
            # For i == 0, context_used remains empty (no previous entries)
            
            # Generate WITH context
            result = call_gemini_with_context(user_message, context_used, mood)
            
            # Store the response
            gemini_responses[i][mood] = {
                "thinking": result["thinking"],
                "response": result["response"],
                "context_used": result.get("context_used", len(context_used))
            }
            
            print(f"✅ (context: {result.get('context_used', len(context_used))})")
            time.sleep(4)
    
    # Second pass: Create training data (now we already have context info)
    print("\n🔨 Second Pass: Building training data...")
    for i, prompt_data in enumerate(prompts):
        user_message = prompt_data['user_message']
        
        # Create training examples for each mood
        for mood in MOODS:
            # Get the response we already generated
            response_data = gemini_responses[i][mood]
            
            # Build context_used for training data
            context_used = []
            if i >= 2:
                for j in range(i-2, i):
                    prev_response = gemini_responses[j][mood]
                    context_used.append({
                        "user": prompts[j]['user_message'],
                        "assistant": prev_response["response"]
                    })
            elif i == 1:
                prev_response = gemini_responses[0][mood]
                context_used.append({
                    "user": prompts[0]['user_message'],
                    "assistant": prev_response["response"]
                })
            
            # Create training example with proper mood-matched context
            training_example = {
                "input": user_message,
                "thinking": response_data["thinking"],
                "output": response_data["response"], 
                "mood": mood,
                "context_used": context_used,
                "context_count": response_data["context_used"]  # Add the count from API response
            }
            
            training_data.append(training_example)
            
            print(f"  ✅ {mood}: {response_data['context_used']} context items")
    
    # Save output
    with open('output.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\n🎉 DONE! Generated {len(training_data)} examples")
    print(f"💾 Saved to output.json")

if __name__ == '__main__':
    main()
    