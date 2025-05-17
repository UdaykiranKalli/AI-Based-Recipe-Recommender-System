import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model  # Importing load_model for consistency

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Embedding, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from PIL import Image

# Paths
data_path = "C:\\Users\\Aarav\\Desktop\\archive\\data\\valid_data.csv"
img_folder = "C:\\Users\\Aarav\\Desktop\\archive\\data\\Valid_Food_Images"
model_save_path = "C:\\Users\\Aarav\\Desktop\\archive\\models\\recipe_model.h5"
tokenizer_save_path = "C:\\Users\\Aarav\\Desktop\\archive\\models\\tokenizer.pkl"

# Load the cleaned dataset
df = pd.read_csv(data_path)

# ✅ Fix: Handle case where dataset is smaller than 50,000
df = df.sample(n=min(50000, len(df)), random_state=42)  # ✅ Use only available rows

# Debug: Print dataset size
print(f"✅ Dataset size after sampling: {len(df)}")

# Tokenize ingredient text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Cleaned_Ingredients'])
sequences = tokenizer.texts_to_sequences(df['Cleaned_Ingredients'])
max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Save tokenizer
os.makedirs(os.path.dirname(tokenizer_save_path), exist_ok=True)
with open(tokenizer_save_path, "wb") as f:
    pickle.dump(tokenizer, f)

# Load images efficiently
def load_image(image_path, target_size=(128, 128)):
    try:
        image = Image.open(image_path)
        image = image.resize(target_size)
        image = np.array(image) / 255.0
        if image.shape == (128, 128, 3):
            return image
    except:
        pass
    return None  # Skip invalid images

images = []
valid_indices = []

for idx, img_name in enumerate(df['Image_Name'].astype(str)):
    img_name = img_name.lower().strip() + ".jpg"
    img_path = os.path.join(img_folder, img_name)

    image = load_image(img_path)
    if image is not None:
        images.append(image)
        valid_indices.append(idx)

# Keep only valid data
df = df.iloc[valid_indices]
padded_sequences = padded_sequences[valid_indices]
images = np.array(images, dtype=np.float32)  # ✅ Use float32 to save memory

# Convert labels to integers (Sparse, no one-hot encoding)
labels = np.array(df.index, dtype=np.int32)  # ✅ Avoid one-hot encoding

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(images, padded_sequences, test_size=0.2, random_state=42)

# Define the model
def build_model(tokenizer, max_len):
    # Image model (CNN)
    image_input = Input(shape=(128, 128, 3))
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_model = Dense(128, activation='relu')(x)

    # Text model (LSTM)
    text_input = Input(shape=(max_len,))
    y = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(text_input)
    y = LSTM(128)(y)
    text_model = Dense(128, activation='relu')(y)

    # Combine models
    combined = concatenate([image_model, text_model])
    z = Dense(64, activation='relu')(combined)
    output = Dense(len(tokenizer.word_index) + 1, activation='softmax')(z)

    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model

# Build the model
model = build_model(tokenizer, max_len)

# Save the model
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)

# ✅ No more one-hot encoding, just use sparse labels
model.fit([X_train, y_train], y_train[:, 0], epochs=10, batch_size=32, validation_split=0.2)




# Save the model
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)

print(f"✅ Model saved to {model_save_path}")
