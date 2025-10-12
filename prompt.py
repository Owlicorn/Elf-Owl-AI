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

    prompt = f"""Generate a response in {mood} mood for this conversation:

{full_prompt}

Return EXACTLY this JSON format:
{{
    "thinking": "Brief thinking process for {mood} mood",
    "response": "Actual response in {mood} mood with emojis"
}}

{mood} mood guidelines:
- Be authentic to {mood} tone
- Use appropriate emojis
- MUST use owl accent(use voice of owl such as ouuuuuu, his-his etc)
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
        "response": f"I'm answering in a {mood} way: {user_message}"
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
    
    # First pass: Generate responses for ALL prompts in ALL moods (no context)
    print("\n🔨 First Pass: Generating responses without context...")
    for i, prompt_data in enumerate(prompts):
        user_message = prompt_data['user_message']
        gemini_responses[i] = {}
        
        print(f"\n📄 Processing: '{user_message}'")
        
        for mood in MOODS:
            print(f"  🎭 {mood}...", end=" ")
            
            # Generate without context first
            result = call_gemini_with_context(user_message, None, mood)
            
            # Store the response
            gemini_responses[i][mood] = {
                "thinking": result["thinking"],
                "response": result["response"]
            }
            
            print("✅")
            time.sleep(4)
    
    # Second pass: Create training data with PROPER mood-matched context
    print("\n🔨 Second Pass: Building training data with mood-matched context...")
    for i, prompt_data in enumerate(prompts):
        user_message = prompt_data['user_message']
        
        # Create training examples for each mood
        for mood in MOODS:
            # Get the response we already generated
            response_data = gemini_responses[i][mood]
            
            # Build context_used with SAME MOOD responses
            context_used = []
            if i >= 1:
                # For prompt at index 1, use prompt 0 responses in SAME MOOD
                prev_response = gemini_responses[i-1][mood]
                context_used.append({
                    "user": prompts[i-1]['user_message'],
                    "assistant": prev_response["response"]
                })
            
            # Create training example with proper mood-matched context
            training_example = {
                "input": user_message,
                "thinking": response_data["thinking"],
                "output": response_data["response"], 
                "mood": mood,
                "context_used": context_used
            }
            
            training_data.append(training_example)
            
            print(f"  ✅ {mood}: {len(context_used)} context items")
    
    # Save output
    with open('output.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\n🎉 DONE! Generated {len(training_data)} examples")
    print(f"💾 Saved to output.json")

if __name__ == '__main__':
    main()