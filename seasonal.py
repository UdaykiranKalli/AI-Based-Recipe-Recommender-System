from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import random
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the recipes from CSV file with better encoding handling
def load_recipes():
    try:
        # Try to find the file in one of several locations
        possible_paths = [
            'data/SeasonalRecipes.csv',
            './data/SeasonalRecipes.csv',
            'D:\\_FinalYear\\cheff\\cheff\\smart-chef-backend\\data\\SeasonalRecipes.csv'
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                print(f"Found CSV file at: {path}")
                break
        
        if not file_path:
            print("CSV file not found in any of the expected locations")
            return pd.DataFrame()
            
        # Try different encodings
        encodings_to_try = ['latin1', 'ISO-8859-1', 'cp1252', 'windows-1252']
        
        for encoding in encodings_to_try:
            try:
                print(f"Attempting to read CSV with {encoding} encoding")
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded {len(df)} recipes with {encoding} encoding")
                return df
            except Exception as e:
                print(f"Error loading CSV with {encoding} encoding: {e}")
                continue
        
        # If we've tried all encodings and none worked, create a sample DataFrame
        print("All encoding attempts failed, creating sample data")
        return create_sample_data()
    except Exception as e:
        print(f"Unexpected error loading CSV: {e}")
        return create_sample_data()

def create_sample_data():
    # Create a simple sample dataset
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6],
        'Name': [
            'Summer Salad', 'Mango Smoothie', 
            'Winter Soup', 'Hot Chocolate', 
            'Rainy Day Stew', 'Spicy Pakora'
        ],
        'Ingredients': [
            'cucumber, tomato, lettuce, olive oil',
            'mango, yogurt, honey',
            'potato, carrot, onion, chicken broth',
            'milk, cocoa powder, sugar',
            'beef, potato, carrot, celery',
            'chickpea flour, onion, chili, water'
        ],
        'Instructions': [
            'Chop vegetables. Mix with olive oil. Serve cold.',
            'Blend all ingredients until smooth.',
            'Dice vegetables. Simmer in broth for 30 minutes.',
            'Heat milk. Stir in cocoa and sugar.',
            'Brown beef. Add vegetables and water. Simmer for 1 hour.',
            'Mix ingredients into batter. Deep fry spoonfuls.'
        ],
        'Season': ['Summer', 'Summer', 'Winter', 'Winter', 'Rainy', 'Rainy'],
        'Image': ['', '', '', '', '', '']
    })

@app.route('/api/seasonal-recipes', methods=['GET'])
def get_seasonal_recipes():
    df = load_recipes()
    
    if df.empty:
        return jsonify({"error": "Could not load recipes"}), 500
    
    # Group recipes by season
    seasons = ['Summer', 'Winter', 'Rainy']
    result = {}
    
    for season in seasons:
        # Filter recipes by season
        season_recipes = df[df['Season'] == season]
        
        # If there are enough recipes for this season, randomly select 4
        if len(season_recipes) >= 4:
            selected_recipes = season_recipes.sample(n=4).to_dict('records')
        else:
            # If not enough recipes, use all available ones
            selected_recipes = season_recipes.to_dict('records')
        
        # Format recipes for frontend
        formatted_recipes = []
        for recipe in selected_recipes:
            # Extract first few ingredients (assuming ingredients are comma-separated)
            ingredients = str(recipe.get('Ingredients', '')).split(',')
            preview_ingredients = ingredients[:2] if len(ingredients) > 2 else ingredients
            
            formatted_recipes.append({
                'id': recipe.get('id', random.randint(1000, 9999)),  # Generate a random ID if none exists
                'title': recipe.get('Name', 'Unnamed Recipe'),
                'preview_ingredients': preview_ingredients,
                'image': recipe.get('Image', ''),  # In case you add image URLs later
                'extendedIngredients': [{'amount': '', 'unit': '', 'name': ing.strip()} for ing in ingredients],
                'instructions': recipe.get('Instructions', ''),
                'season': recipe.get('Season', '')
            })
        
        result[season.lower()] = formatted_recipes
    
    return jsonify(result)

# Add a simple health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0')  # Use 0.0.0.0 to make it accessible from other devices