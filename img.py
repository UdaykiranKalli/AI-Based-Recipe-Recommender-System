from flask import Flask, send_from_directory, abort
import os

app = Flask(__name__)  # Define Flask app first

# Corrected route
IMAGE_FOLDER = os.path.join(os.getcwd(), "data", "Valid_Food_Images")

@app.route('/images/<filename>')  # This should be a relative path, not absolute
def get_image(filename):
    img_path = os.path.join(IMAGE_FOLDER, filename)

    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return abort(404, description="Image not found")

    print(f"✅ Serving Image: {img_path}")
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
