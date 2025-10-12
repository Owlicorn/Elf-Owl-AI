from flask import Flask, render_template, request, jsonify, session
import os
import json
from datetime import datetime
from inference import ElfOwlInference
import config

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'elf-owl-secret-key-2024')
app.config['SESSION_TYPE'] = 'filesystem'

# Initialize Elf Owl AI
try:
    elf_owl = ElfOwlInference()
    ai_ready = True
    print("✅ Elf Owl AI initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize Elf Owl AI: {e}")
    elf_owl = None
    ai_ready = False

def save_user_prompt(user_message):
    """Save ONLY user prompts to prompt.json - simple and clean"""
    try:
        prompt_data = {
            "user_message": user_message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create file if it doesn't exist
        if not os.path.exists('prompt.json'):
            with open('prompt.json', 'w') as f:
                json.dump([], f)
        
        # Read existing data
        with open('prompt.json', 'r') as f:
            existing_data = json.load(f)
        
        # Append new prompt
        existing_data.append(prompt_data)
        
        # Write back to file
        with open('prompt.json', 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        print(f"✅ Saved user prompt: '{user_message}'")
        
    except Exception as e:
        print(f"❌ Error saving prompt: {e}")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', ai_ready=ai_ready)

@app.route('/chat')
def chat():
    """Chat interface"""
    return render_template('chat.html')

@app.route('/api/moods', methods=['GET'])
def api_get_moods():
    """Get all available moods"""
    try:
        moods = elf_owl.get_available_moods()
        return jsonify({
            'moods': moods,
            'count': len(moods),
            'default_mood': session.get('preferred_mood', 'auto')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mood/set', methods=['POST'])
def api_set_mood():
    """Set mood for current session or specific request"""
    try:
        data = request.get_json()
        mood = data.get('mood', '').strip().lower()
        
        if not mood:
            return jsonify({'error': 'No mood provided'}), 400
        
        if mood == 'auto':
            # Use automatic mood detection
            session.pop('preferred_mood', None)
            return jsonify({
                'status': 'success', 
                'message': 'Mood set to automatic detection',
                'mood': 'auto'
            })
        
        # Check if mood is valid
        available_moods = elf_owl.get_available_moods()
        if mood not in available_moods:
            return jsonify({
                'error': f'Invalid mood. Available moods: {", ".join(available_moods.keys())}'
            }), 400
        
        # Set mood for session
        session['preferred_mood'] = mood
        
        return jsonify({
            'status': 'success',
            'message': f'Mood set to {mood}',
            'mood': mood,
            'description': available_moods[mood]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat API endpoint with mood support"""
    if not ai_ready:
        return jsonify({'error': 'Elf Owl AI is not ready'}), 503
    
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        forced_mood = data.get('mood', '').strip().lower()
        
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get session preferred mood if not specified in request
        if not forced_mood:
            forced_mood = session.get('preferred_mood')
        
        conversation_history = session.get('conversation', [])
        
        # Save ONLY the user message to prompt.json
        save_user_prompt(message)
        
        # Generate response
        response_data = elf_owl.generate_response(
            message, 
            conversation_history,
            forced_mood=forced_mood if forced_mood and forced_mood != 'auto' else None
        )
        
        # Update conversation history
        conversation_history.append({
            'user': message,
            'assistant': response_data['response'],
            'mood': response_data['mood'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 messages in session
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        session['conversation'] = conversation_history
        
        return jsonify({
            'response': response_data['response'],
            'thinking': response_data.get('thinking', ''),
            'mood': response_data['mood'],
            'mood_source': response_data.get('mood_source', 'auto_detected'),
            'context_used': response_data.get('context_used', False),
            'tokens_generated': response_data.get('tokens_generated', 0),
            'history_length': len(conversation_history),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Direct generation API"""
    if not ai_ready:
        return jsonify({'error': 'Elf Owl AI is not ready'}), 503
    
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        max_length = data.get('max_length', config.Config.MAX_GENERATION_LENGTH)
        temperature = data.get('temperature', config.Config.TEMPERATURE)
        
        if not prompt:
            return jsonify({'error': 'Empty prompt'}), 400
        
        # Save the prompt
        save_user_prompt(prompt)
        
        response_data = elf_owl.generate_response(
            prompt, 
            max_length=max_length,
            temperature=temperature
        )
        
        return jsonify({
            'response': response_data['response'],
            'thinking': response_data.get('thinking', ''),
            'mood': response_data.get('mood', 'neutral'),
            'tokens_generated': response_data.get('tokens_generated', 0),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/clear', methods=['POST'])
def api_clear():
    """Clear conversation history"""
    session.pop('conversation', None)
    return jsonify({'status': 'success', 'message': 'Conversation history cleared'})

@app.route('/api/info', methods=['GET'])
def api_info():
    """Get AI system information"""
    return jsonify({
        'name': 'Elf Owl AI',
        'version': '2.0.0',
        'description': 'Auto-scaling AI with free generation and mood system',
        'model_config': {
            'd_model': config.Config.D_MODEL,
            'n_layers': config.Config.N_LAYERS,
            'n_heads': config.Config.N_HEADS,
            'vocab_size': config.Config.VOCAB_SIZE
        },
        'generation_config': {
            'min_tokens': config.Config.MIN_GENERATION_LENGTH,
            'max_tokens': config.Config.MAX_GENERATION_LENGTH,
            'temperature': config.Config.TEMPERATURE,
            'top_k': config.Config.TOP_K,
            'top_p': config.Config.TOP_P
        },
        'moods_available': config.Config.MOODS,
        'data_sources': ['lang_model.txt', 'MongoDB'],
        'free_generation': True
    })

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ready' if ai_ready else 'not ready',
        'model_loaded': ai_ready,
        'service': 'Elf Owl AI',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get training data statistics"""
    try:
        from data_loader import DataLoader
        data_loader = DataLoader()
        stats = data_loader.get_data_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Documentation
@app.route('/api/docs')
def api_docs():
    """API Documentation"""
    docs = {
        'endpoints': {
            '/api/chat': {
                'method': 'POST',
                'description': 'Chat with Elf Owl AI',
                'parameters': {
                    'message': 'string (required) - User message',
                    'max_length': 'integer (optional) - Max tokens to generate',
                    'temperature': 'float (optional) - Generation temperature'
                }
            },
            '/api/generate': {
                'method': 'POST', 
                'description': 'Direct text generation',
                'parameters': {
                    'prompt': 'string (required) - Input prompt',
                    'max_length': 'integer (optional) - Max tokens to generate',
                    'temperature': 'float (optional) - Generation temperature'
                }
            },
            '/api/clear': {
                'method': 'POST',
                'description': 'Clear conversation history'
            },
            '/api/info': {
                'method': 'GET',
                'description': 'Get system information'
            },
            '/api/health': {
                'method': 'GET', 
                'description': 'Health check'
            },
            '/api/stats': {
                'method': 'GET',
                'description': 'Get training data statistics'
            }
        },
        'features': [
            'Auto-scaling model architecture',
            'Free generation (1-512+ tokens)',
            'Mood-based responses',
            'Dual data sources (lang_model.txt + MongoDB)',
            'Conversation context awareness',
            'Thinking process visualization'
        ]
    }
    return jsonify(docs)

if __name__ == '__main__':
    print("🚀 Starting Elf Owl AI Server...")
    print(f"🌐 API Documentation: http://{config.Config.API_HOST}:{config.Config.API_PORT}/api/docs")
    print(f"💬 Chat Interface: http://{config.Config.API_HOST}:{config.Config.API_PORT}/chat")
    print("🎯 Free Generation Mode: ENABLED")
    print("📝 User prompts saving: ENABLED (prompt.json)")
    
    app.run(
        host=config.Config.API_HOST,
        port=config.Config.API_PORT,
        debug=config.Config.DEBUG
    )