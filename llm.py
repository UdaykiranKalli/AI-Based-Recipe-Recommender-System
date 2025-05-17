from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import json
from googletrans import Translator
import speech_recognition as sr
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pyttsx3
import io
import base64
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Translator
translator = Translator()

# Function to translate text
def translate_text(text, target_language):
    if target_language == "English":
        return text
    
    language_code_map = {"Telugu": "te", "Hindi": "hi"}
    target_code = language_code_map.get(target_language, "en")
    
    try:
        result = translator.translate(text, dest=target_code)
        return result.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

# Initialize the local LLM
def initialize_llm(model_path="D:\_FinalYear\cheff\cheff\smart-chef-backend\models\llama-2-7b-chat.Q5_K_M.gguf"):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        max_tokens=4000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm

# Load the LLM
try:
    llm = initialize_llm()
    llm_loaded = True
except Exception as e:
    print(f"Error loading LLM: {e}")
    llm_loaded = False

# Function to generate the recipe
def generate_recipe_from_ingredients(ingredients, recipe_type, language="English", persons=2):
    prompt = f"""
    Generate a detailed recipe for {persons} persons using only these ingredients: {ingredients}.
    Recipe type: {recipe_type}.
    
    The response should include:
    1. A creative recipe name
    2. Ingredients list with measurements for {persons} persons
    3. Step-by-step cooking instructions must not be too long
    4. Approximate cooking time
    5. Nutritional information (estimated)
    """
    
    if llm_loaded:
        recipe = llm.invoke(prompt)
        
        if language != "English":
            recipe = translate_text(recipe, language)
        
        # Trim everything before the recipe name
        lines = recipe.split('\n')
        recipe_name_index = -1
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['recipe', 'creative', 'name:', 'title:']):
                # Find the first non-empty line starting with the recipe name
                for j in range(i, len(lines)):
                    if lines[j].strip() and not lines[j].strip().startswith(('*', '-', '1.', '2.', '3.')):
                        recipe_name_index = j
                        break
                break
        
        if recipe_name_index != -1:
            lines = lines[recipe_name_index:]
        
        # Parse the recipe text to extract components
        title = lines[0] if lines else "Recipe"
        
        # Extract ingredients
        ingredients_section = []
        instructions_section = []
        
        in_ingredients = False
        in_instructions = False
        
        for line in lines:
            if "Ingredients" in line or "ingredients" in line:
                in_ingredients = True
                in_instructions = False
                continue
            elif "Instructions" in line or "instructions" in line or "Steps" in line or "Directions" in line:
                in_ingredients = False
                in_instructions = True
                continue
            elif "Cooking time" in line or "Nutritional" in line:
                in_ingredients = False
                in_instructions = False
            
            if in_ingredients and line.strip():
                ingredients_section.append(line.strip())
            elif in_instructions and line.strip():
                instructions_section.append(line.strip())
        
        # Format the recipe data
        recipe_data = {
            "title": title.replace("Recipe:", "").strip(),
            "ingredients": ingredients_section,
            "instructions": " ".join(instructions_section),
            "full_text": "\n".join(lines)
        }
        
        return recipe_data
    else:
        return {"error": "LLM not loaded. Please check your model configuration."}

# Function to recognize speech from audio file
def recognize_speech_from_file(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Speech recognition could not understand audio"
        except sr.RequestError:
            return "Could not request results from speech recognition service"

# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    engine.save_to_file(text, temp_filename)
    engine.runAndWait()
    
    # Read the file content and encode as base64
    with open(temp_filename, 'rb') as audio_file:
        audio_data = audio_file.read()
        base64_audio = base64.b64encode(audio_data).decode('utf-8')
    
    # Clean up the temporary file
    os.remove(temp_filename)
    
    return base64_audio

# API Routes
@app.route('/api/recommend-recipe', methods=['POST'])
def recommend_recipe():
    data = request.json
    
    ingredients = data.get('ingredients', '')
    recipe_type = data.get('mealType', 'Lunch or Dinner')
    language = data.get('language', 'English')
    persons = data.get('persons', 2)
    
    if not ingredients or len(ingredients.split(',')) < 3:
        return jsonify({"error": "Please provide at least 3 ingredients"}), 400
    
    try:
        recipe = generate_recipe_from_ingredients(ingredients, recipe_type, language, persons)
        return jsonify({"recipe": recipe})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    # Save the audio file temporarily
    temp_dir = tempfile.gettempdir()
    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(temp_dir, filename)
    audio_file.save(filepath)
    
    try:
        text = recognize_speech_from_file(filepath)
        os.remove(filepath)  # Clean up
        return jsonify({"text": text})
    except Exception as e:
        os.remove(filepath)  # Clean up
        return jsonify({"error": str(e)}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def get_speech():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        base64_audio = text_to_speech(text)
        return jsonify({"audio": base64_audio})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/random-recipe', methods=['GET'])
def random_recipe():
    language = request.args.get('language', 'English')
    persons = int(request.args.get('persons', 2))
    
    # List of common ingredients for a random recipe
    common_ingredients = [
        "chicken, rice, onion, garlic, olive oil, salt, pepper",
        "pasta, tomato, basil, cheese, olive oil, garlic",
        "potato, butter, milk, salt, pepper, cheese",
        "eggs, flour, sugar, milk, vanilla extract",
        "ground beef, onion, garlic, tomato, spices"
    ]
    
    import random
    selected_ingredients = random.choice(common_ingredients)
    recipe_types = ["Breakfast", "Lunch or Dinner", "Dessert"]
    selected_type = random.choice(recipe_types)
    
    try:
        recipe = generate_recipe_from_ingredients(selected_ingredients, selected_type, language, persons)
        return jsonify({"recipe": recipe})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/image-recipe', methods=['POST'])
def image_recipe():
    # This is a placeholder for image processing
    # In a real implementation, you'd use a vision model to identify ingredients
    return jsonify({
        "recipe": {
            "title": "Recipe from Image",
            "ingredients": ["This is a placeholder for image-based recipe generation", 
                           "Currently not implemented in this API"],
            "instructions": "Image-based recipe generation would require a computer vision model to identify ingredients."
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)