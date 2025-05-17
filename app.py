# app_api.py
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import random
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

DATASET_PATH = "D:\\_FinalYear\\cheff\\cheff\\smart-chef-backend\\data\\valid_data.csv"
MODEL_PATH = "D:\\_FinalYear\\cheff\\cheff\\smart-chef-backend\\models\\recipe_model.h5"
TOKENIZER_PATH = "D:\_FinalYear\cheff\cheff\smart-chef-backend\models\tokenizer.pkl"
CLASS_NAMES_PATH = "D:\\_FinalYear\\cheff\\cheff\\smart-chef-backend\\class_names.json"
IMAGE_FOLDER = "D:\\_FinalYear\\cheff\\cheff\\smart-chef-backend\\data\\Valid_Food_Images"
MAX_LEN = 407  # Must match training parameter


# Language mapping
LANGUAGE_MAPPING = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi"
}

# Recipe categories for better filtering
RECIPE_CATEGORIES = {
    "Dessert": ['cake', 'cookie', 'sweet', 'dessert', 'custard', 'pudding', 'tart', 'pie', 'ice cream', 'chocolate'],
    "Breakfast": ['breakfast', 'morning', 'toast', 'pancake', 'waffle', 'cereal', 'oatmeal', 'muffin'],
    "Beverage": ['drink', 'smoothie', 'shake', 'milk', 'coffee', 'tea', 'beverage', 'juice'],
    "Main Course": ['dinner', 'lunch', 'entree', 'main', 'steak', 'roast', 'curry', 'pasta'],
    "Appetizer": ['appetizer', 'starter', 'snack', 'dip', 'finger food']
}

# Common ingredient combinations and their likely recipe types
INGREDIENT_COMBINATIONS = {
    frozenset(['milk', 'sugar']): ['pudding', 'custard', 'ice cream', 'shake', 'smoothie'],
    frozenset(['milk', 'almond']): ['almond milk', 'shake', 'pudding', 'smoothie'],
    frozenset(['milk', 'almond', 'sugar']): ['almond milk dessert', 'pudding', 'custard', 'sweet'],
    frozenset(['flour', 'sugar', 'butter']): ['cake', 'cookie', 'pastry', 'bread'],
    frozenset(['tomato', 'onion', 'garlic']): ['sauce', 'curry', 'soup', 'stew'],
    frozenset(['chicken', 'onion']): ['curry', 'roast', 'stew', 'soup'],
    frozenset(['rice', 'vegetable']): ['fried rice', 'pilaf', 'biryani', 'bowl'],
    # Add more common combinations as needed
}

# Load dataset
def load_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
        required_columns = ["Title", "Ingredients", "Instructions", "Image_Name"]
        
        if not all(col in df.columns for col in required_columns):
            print("Error: Missing required columns in dataset")
            return None
            
        # Clean the data
        # 1. Drop rows with missing title or ingredients
        df = df.dropna(subset=["Title", "Ingredients"])
        
        # 2. Convert any numeric values to strings in text columns
        for col in ["Title", "Ingredients", "Instructions"]:
            df[col] = df[col].astype(str)
            
        # 3. Fill any remaining NaN values in Image_Name with a default
        df["Image_Name"] = df["Image_Name"].fillna("default")
        
        print(f"Loaded {len(df)} valid recipes from dataset")
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

# Load model
def load_recipe_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Load tokenizer
def load_tokenizer():
    try:
        with open(TOKENIZER_PATH, 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        return None

# Fix Image Path Issues
def get_image_path(image_name):
    if "." in image_name:
        img_path = os.path.join(IMAGE_FOLDER, image_name)
        if os.path.exists(img_path):
            return img_path

    supported_extensions = [".jpg", ".jpeg", ".png"]
    for ext in supported_extensions:
        img_path = os.path.join(IMAGE_FOLDER, f"{image_name}{ext}")
        if os.path.exists(img_path):
            return img_path

    return None  # Return None if no matching file is found

# Load dataset, model and tokenizer at startup
df = load_dataset()
model = load_recipe_model()
tokenizer = load_tokenizer()

# Load class names
try:
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
except Exception as e:
    print(f"Error loading class names: {str(e)}")
    class_names = []

# Helper function to infer recipe category from title and ingredients
def infer_recipe_category(title, ingredients):
    # Check if inputs are strings
    if not isinstance(title, str):
        title = str(title) if title is not None else ""
    if not isinstance(ingredients, str):
        ingredients = str(ingredients) if ingredients is not None else ""
        
    title_lower = title.lower()
    ingredients_lower = ingredients.lower()
    
    for category, keywords in RECIPE_CATEGORIES.items():
        if any(keyword in title_lower for keyword in keywords):
            return category
    
    # If no category found in title, check ingredients
    dessert_ingredients = ['sugar', 'chocolate', 'vanilla', 'honey', 'caramel']
    if any(ing in ingredients_lower for ing in dessert_ingredients):
        return "Dessert"
    
    protein_ingredients = ['chicken', 'beef', 'pork', 'fish', 'shrimp', 'tofu']
    if any(ing in ingredients_lower for ing in protein_ingredients):
        return "Main Course"
    
    # Default category
    return "Other"

# Significantly improved ingredient handling for better recipe recommendations
def predict_recipes_from_ingredients(ingredients_text, meal_type='', top_n=5):
    """
    Enhanced function to predict recipes from ingredients with improved relevance.
    """
    # Parse ingredients into a list, normalize to lowercase and trim
    ingredient_list = [ing.strip().lower() for ing in ingredients_text.split(',')]
    
    # Create a frozen set for ingredient combination matching
    ingredient_set = frozenset(ingredient_list)
    
    # Score recipes based on comprehensive ingredient matching
    scored_recipes = []
    for _, recipe in df.iterrows():
        # Check if Title is a float or NaN value and handle it
        if pd.isna(recipe["Title"]) or not isinstance(recipe["Title"], str):
            continue  # Skip this recipe if Title is not a string
            
        recipe_title = recipe["Title"].lower()
        
        # Check if Ingredients is a float or NaN value and handle it
        if pd.isna(recipe["Ingredients"]) or not isinstance(recipe["Ingredients"], str):
            continue  # Skip this recipe if Ingredients is not a string
            
        recipe_ingredients_text = recipe["Ingredients"].lower()
        recipe_ingredients_list = [ing.strip().lower() for ing in recipe["Ingredients"].split(',')]
        
        # Initialize scoring metrics
        match_score = 0
        
        # 1. Count basic matches (how many input ingredients appear in recipe)
        match_count = sum(1 for ing in ingredient_list if ing in recipe_ingredients_text)
        if match_count == 0:
            continue  # Skip recipes with no matching ingredients
        
        # 2. Calculate match percentage (input ingredients used in recipe)
        match_percentage = match_count / len(ingredient_list) if ingredient_list else 0
        match_score += match_percentage * 2  # Base score
        
        # 3. Bonus for using ALL ingredients (complete match)
        if match_count == len(ingredient_list):
            match_score += 1.5  # Significant bonus for using all ingredients
        
        # 4. Check for exact matches vs partial matches
        exact_matches = sum(1 for ing in ingredient_list if ing in recipe_ingredients_list)
        match_score += (exact_matches * 0.2)  # Bonus for exact matches
        
        # 5. Check ingredient prominence (ingredients appearing early in the list)
        prominence_score = 0
        for i, ing in enumerate(recipe_ingredients_list[:5]):  # Check first 5 ingredients
            if any(user_ing in ing for user_ing in ingredient_list):
                prominence_score += (5 - i) * 0.05  # Higher score for earlier matches
        match_score += prominence_score
        
        # 6. Check for known ingredient combinations
        for combo, related_dishes in INGREDIENT_COMBINATIONS.items():
            if ingredient_set.issubset(combo) or combo.issubset(ingredient_set):
                # The input ingredients match a known combination
                if any(dish in recipe_title for dish in related_dishes):
                    match_score += 0.5  # Bonus for matching a recommended dish type
        
        # 7. Apply meal type filtering
        category_match = True
        if meal_type:
            recipe_category = infer_recipe_category(recipe_title, recipe_ingredients_text)
            
            if meal_type == "Breakfast" and recipe_category != "Breakfast":
                category_match = False
            elif meal_type == "Lunch or Dinner" and recipe_category not in ["Main Course", "Appetizer"]:
                category_match = False
            elif meal_type == "Dessert" and recipe_category != "Dessert":
                category_match = False
            
            # Bonus for category match
            if category_match:
                match_score += 0.3
        
        if category_match:
            # Calculate what percentage of recipe ingredients are covered by user ingredients
            recipe_coverage = match_count / len(recipe_ingredients_list) if recipe_ingredients_list else 0
            
            # Store detailed matching info for UI feedback
            unused_ingredients = [ing for ing in ingredient_list 
                                if ing not in recipe_ingredients_text]
            
            scored_recipes.append({
                "recipe": recipe,
                "match_score": match_score,
                "match_count": match_count,
                "match_percentage": match_percentage * 100,  # Convert to percentage
                "recipe_coverage": recipe_coverage * 100,  # Convert to percentage
                "unused_ingredients": unused_ingredients
            })
    
    # Sort by match score (highest first)
    scored_recipes.sort(key=lambda x: x["match_score"], reverse=True)
    
    # Try to use ML model if available for additional recommendations
    model_recipes = []
    if model is not None and tokenizer is not None and len(scored_recipes) < top_n:
        try:
            # Tokenize and pad the ingredients text
            sequences = tokenizer.texts_to_sequences([ingredients_text])
            padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)
            
            # Create a dummy image input (since our model expects both inputs)
            dummy_image = np.zeros((1, 128, 128, 3))
            
            # Get model predictions
            predictions = model.predict([dummy_image, padded_sequences])[0]
            
            # Get indices of top predictions
            top_indices = np.argsort(predictions)[::-1][:top_n * 3]
            predicted_classes = [class_names[i] for i in top_indices]
            
            # Find matching recipes in the dataframe
            for food_class in predicted_classes:
                matching_recipes = df[df["Title"].str.contains(food_class, case=False, na=False)]
                
                if not matching_recipes.empty:
                    # Apply meal type filter if provided
                    for _, recipe in matching_recipes.iterrows():
                        # Check for non-string values in Ingredients
                        if pd.isna(recipe["Ingredients"]) or not isinstance(recipe["Ingredients"], str):
                            continue
                            
                        recipe_ingredients_text = recipe["Ingredients"].lower()
                        match_count = sum(1 for ing in ingredient_list if ing in recipe_ingredients_text)
                        
                        # Only include if it uses at least one of our ingredients
                        if match_count > 0:
                            # Check if this recipe is already in scored_recipes
                            if not any(r["recipe"]["Title"] == recipe["Title"] for r in scored_recipes):
                                # Calculate match percentage for consistency
                                match_percentage = match_count / len(ingredient_list) if ingredient_list else 0
                                recipe_ingredients_list = [ing.strip().lower() for ing in recipe["Ingredients"].split(',')]
                                recipe_coverage = match_count / len(recipe_ingredients_list) if recipe_ingredients_list else 0
                                
                                unused_ingredients = [ing for ing in ingredient_list 
                                                    if ing not in recipe_ingredients_text]
                                
                                model_recipes.append({
                                    "recipe": recipe,
                                    "match_score": match_percentage * 1.5,  # Base score
                                    "match_count": match_count,
                                    "match_percentage": match_percentage * 100,
                                    "recipe_coverage": recipe_coverage * 100,
                                    "unused_ingredients": unused_ingredients,
                                    "model_suggested": True
                                })
                
                # Stop when we have enough recipes
                if len(model_recipes) >= top_n - len(scored_recipes):
                    break
            
            # Sort model recipes by score
            model_recipes.sort(key=lambda x: x["match_score"], reverse=True)
        except Exception as e:
            print(f"Model prediction failed: {str(e)}")
    
    # Combine scored recipes with model recipes (if any)
    combined_recipes = scored_recipes + model_recipes
    
    # Return the top N recipes
    return combined_recipes[:top_n]

# Modified recipe recommendation endpoint with enhanced matching
@app.route('/api/recommend-recipes', methods=['POST'])
def recommend_recipes():
    data = request.json
    ingredients_input = data.get('ingredients', '')
    serving_size = int(data.get('persons', 1))
    language = data.get('language', 'English')
    meal_type = data.get('mealType', '')
    count = int(data.get('count', 4))  # Default to 4 recipes
    
    # Limit maximum recipes to 10 for performance
    count = min(count, 10)

    if not ingredients_input:
        return jsonify({'error': 'No ingredients provided'}), 400
    
    # Get enhanced recipe predictions with detailed scoring
    recipe_matches = predict_recipes_from_ingredients(ingredients_input, meal_type, top_n=count)
    
    if not recipe_matches:
        return jsonify({'error': 'No recipes found with these ingredients'}), 404
    
    recipes = []
    for recipe_data in recipe_matches:
        recipe = recipe_data["recipe"]
        img_path = get_image_path(recipe["Image_Name"])
        img_url = f"http://localhost:5000/images/{recipe['Image_Name']}" if img_path else "/default-food.jpg"

        # Ensure ingredients is a string before splitting
        if not isinstance(recipe["Ingredients"], str):
            recipe["Ingredients"] = str(recipe["Ingredients"])
            
        # Convert ingredients to list for better display
        ingredients_list = [ing.strip() for ing in recipe["Ingredients"].split(',')]
        
        # Get user ingredients for matching display
        user_ingredients = [ing.strip().lower() for ing in ingredients_input.split(',')]
        matching_ingredients = [ing for ing in ingredients_list 
                               if any(user_ing in ing.lower() for user_ing in user_ingredients)]
        
        season = get_recipe_season(recipe["Title"], recipe["Ingredients"])
        category = infer_recipe_category(recipe["Title"], recipe["Ingredients"])
        
        # Ensure all fields are valid strings
        title = str(recipe["Title"]) if recipe["Title"] is not None else "Untitled Recipe"
        instructions = str(recipe["Instructions"]) if recipe["Instructions"] is not None else "No instructions available"
        
        # Enhanced recipe object with match information
        recipe_obj = {
            'title': title,
            'image': img_url,
            'ingredients': ingredients_list,
            'preview_ingredients': ingredients_list[:3],  # First 3 ingredients for preview
            'matching_ingredients': matching_ingredients,  # Show which ingredients matched
            'unused_ingredients': recipe_data["unused_ingredients"],
            'match_percentage': round(recipe_data["match_percentage"], 1),
            'recipe_coverage': round(recipe_data["recipe_coverage"], 1),
            'instructions': instructions,
            'description': f"A delicious {title} recipe using your ingredients.",
            'season': season,
            'category': category,
            'model_suggested': recipe_data.get("model_suggested", False)
        }
        recipes.append(recipe_obj)
    
    response = {
        'recipes': recipes,
        'query_ingredients': ingredients_input.split(','),
        'ingredient_count': len(ingredients_input.split(','))
    }

    return jsonify(response)

# Updated single recipe endpoint with enhanced matching
@app.route('/api/recommend-recipe', methods=['POST'])
def recommend_recipe():
    data = request.json
    ingredients_input = data.get('ingredients', '')
    serving_size = int(data.get('persons', 1))
    language = data.get('language', 'English')
    meal_type = data.get('mealType', '')

    if not ingredients_input:
        return jsonify({'error': 'No ingredients provided'}), 400
    
    # Get the best matching recipe using the enhanced prediction function
    recipe_matches = predict_recipes_from_ingredients(ingredients_input, meal_type, top_n=1)
    
    if not recipe_matches:
        return jsonify({'error': 'No recipes found with these ingredients'}), 404
    
    recipe_data = recipe_matches[0]
    recipe = recipe_data["recipe"]
    
    img_path = get_image_path(recipe["Image_Name"])
    img_url = f"http://localhost:5000/images/{recipe['Image_Name']}" if img_path else "/default-food.jpg"

    # Ensure ingredients is a string before splitting
    if not isinstance(recipe["Ingredients"], str):
        recipe["Ingredients"] = str(recipe["Ingredients"])
        
    # Convert ingredients to list for better display
    ingredients_list = [ing.strip() for ing in recipe["Ingredients"].split(',')]
    
    # Get user ingredients for matching display
    user_ingredients = [ing.strip().lower() for ing in ingredients_input.split(',')]
    matching_ingredients = [ing for ing in ingredients_list 
                           if any(user_ing in ing.lower() for user_ing in user_ingredients)]
    
    season = get_recipe_season(recipe["Title"], recipe["Ingredients"])
    category = infer_recipe_category(recipe["Title"], recipe["Ingredients"])
    
    # Ensure all fields are valid strings
    title = str(recipe["Title"]) if recipe["Title"] is not None else "Untitled Recipe"
    instructions = str(recipe["Instructions"]) if recipe["Instructions"] is not None else "No instructions available"
    
    response = {
        'title': title,
        'image': img_url,
        'ingredients': ingredients_list,
        'matching_ingredients': matching_ingredients,  # Show which ingredients matched
        'unused_ingredients': recipe_data["unused_ingredients"],
        'match_percentage': round(recipe_data["match_percentage"], 1),
        'recipe_coverage': round(recipe_data["recipe_coverage"], 1),
        'instructions': instructions,
        'description': f"A delicious {title} recipe using your ingredients.",
        'season': season,
        'category': category,
        'model_suggested': recipe_data.get("model_suggested", False)
    }

    return jsonify({'recipe': response})

# Helper function to determine recipe season based on ingredients
def get_recipe_season(title, ingredients):
    # Check if inputs are strings
    if not isinstance(title, str):
        title = str(title) if title is not None else ""
    if not isinstance(ingredients, str):
        ingredients = str(ingredients) if ingredients is not None else ""
        
    title_lower = title.lower()
    ingredients_lower = ingredients.lower()
    
    # Summer ingredients and keywords
    summer_keywords = ['summer', 'grill', 'barbecue', 'bbq', 'watermelon', 'berries', 
                      'tomato', 'cucumber', 'zucchini', 'eggplant', 'corn', 'peach', 
                      'mango', 'ice cream', 'cold']
    
    # Winter ingredients and keywords
    winter_keywords = ['winter', 'stew', 'soup', 'roast', 'baked', 'potato', 'squash', 
                      'pumpkin', 'cinnamon', 'nutmeg', 'clove', 'ginger', 'hot', 'warm']
    
    # Check for summer keywords
    for keyword in summer_keywords:
        if keyword in title_lower or keyword in ingredients_lower:
            return "Summer"
    
    # Check for winter keywords
    for keyword in winter_keywords:
        if keyword in title_lower or keyword in ingredients_lower:
            return "Winter"
    
    # Default - all season
    return "All Season"

# Serve Images
@app.route('/images/<filename>')
def get_image(filename):
    img_path = get_image_path(filename)

    if not img_path:
        print(f"❌ Image not found: {filename}")
        return abort(404, description="Image not found")

    print(f"✅ Serving Image: {img_path}")
    return send_file(img_path, mimetype='image/jpeg')

# Random Recipe API
@app.route('/api/random-recipe', methods=['GET'])
def random_recipe():
    serving_size = int(request.args.get('persons', 1))
    language = request.args.get('language', 'English')

    if df is not None and not df.empty:
        recipe = df.sample(1).iloc[0]

        # Ensure all fields are strings
        title = str(recipe["Title"]) if recipe["Title"] is not None else "Untitled Recipe"
        ingredients = str(recipe["Ingredients"]) if recipe["Ingredients"] is not None else "No ingredients available"
        instructions = str(recipe["Instructions"]) if recipe["Instructions"] is not None else "No instructions available"
        
        img_path = get_image_path(recipe["Image_Name"])
        img_url = f"http://localhost:5000/images/{recipe['Image_Name']}" if img_path else "/default-food.jpg"

        response = {
            'title': title,
            'image': img_url,
            'ingredients': [ing.strip() for ing in ingredients.split(',')],  # Convert to list
            'instructions': instructions,
            'description': f"A delicious {title} recipe."
        }

        return jsonify({'recipe': response})
    else:
        return jsonify({'error': 'No recipes available'}), 404

# Image-Based Recipe Search
@app.route('/api/image-recipe', methods=['POST'])
def image_recipe():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    serving_size = int(request.form.get('persons', 1))
    language = request.form.get('language', 'English')

    try:
        img = Image.open(file)
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        dummy_text = np.zeros((1, MAX_LEN))
        prediction = model.predict([img_array, dummy_text])
        predicted_class = class_names[np.argmax(prediction[0])]

        filtered_df = df[df["Title"].str.contains(predicted_class, case=False, na=False)]

        if filtered_df.empty:
            return jsonify({'error': f'No recipes found for detected food: {predicted_class}'}), 404

        recipe = filtered_df.iloc[0]

        # Ensure all fields are strings
        title = str(recipe["Title"]) if recipe["Title"] is not None else "Untitled Recipe"
        ingredients = str(recipe["Ingredients"]) if recipe["Ingredients"] is not None else "No ingredients available"
        instructions = str(recipe["Instructions"]) if recipe["Instructions"] is not None else "No instructions available"
        
        img_path = get_image_path(recipe["Image_Name"])
        img_url = f"http://localhost:5000/images/{recipe['Image_Name']}" if img_path else "/default-food.jpg"

        response = {
            'title': title,
            'image': img_url,
            'ingredients': [ing.strip() for ing in ingredients.split(',')],  # Convert to list
            'instructions': instructions,
            'description': f"A delicious {title} recipe.",
            'detected_food': predicted_class
        }

        return jsonify({'recipe': response})
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

# Add a utility endpoint to get recipe categories
@app.route('/api/recipe-categories', methods=['GET'])
def get_recipe_categories():
    return jsonify({'categories': list(RECIPE_CATEGORIES.keys())})

# Add a utility endpoint to get common ingredient combinations
@app.route('/api/ingredient-combinations', methods=['GET'])
def get_ingredient_combinations():
    combinations = [
        {
            'ingredients': list(combo),
            'suggested_recipes': recipes
        }
        for combo, recipes in INGREDIENT_COMBINATIONS.items()
    ]
    return jsonify({'combinations': combinations})

if __name__ == '__main__':
    app.run(debug=True, port=5000)