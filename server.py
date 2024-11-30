from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import random
import json
import mysql.connector
import os
from werkzeug.utils import secure_filename
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from entity_extraction import extract_entities

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # This is to enable CORS (Cross-Origin Resource Sharing) for frontend

# Configure file upload settings
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

# Ensure uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# MySQL configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",  # Replace with your MySQL username
    password="",  # Replace with your MySQL password
    database="snackus"
)

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to add a new restaurant (with image upload)
@app.route('/add_restaurant', methods=['POST'])
def add_restaurant():
    try:
        # Get data from the request
        data = request.form
        title = data['title']
        restaurant_name = data['restaurant_name']
        rating = data['rating']
        location = data['location']
        sub_location = data['sub_location']
        special_item = data['special_item']
        description = data['description']
        recommendation = data['recommendation']
        category = data['category']  # Get the category from the request

        # Validate category
        if category not in ['local', 'mid-range', 'high-end']:
            return jsonify({"error": "Invalid category. Allowed values are 'local', 'mid-range', or 'high-end'."}), 400

        # Handle image uploads
        restaurant_photo = request.files['restaurant_photo']
        menu_photo = request.files['menu_photo']
        
        photo_url = None
        menu_photo_url = None

        if restaurant_photo and allowed_file(restaurant_photo.filename):
            filename = secure_filename(restaurant_photo.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Use correct path
            restaurant_photo.save(file_path)
            photo_url = f"http://localhost:5000/uploads/{filename}"  # Correct URL

        if menu_photo and allowed_file(menu_photo.filename):
            filename = secure_filename(menu_photo.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Use correct path
            menu_photo.save(file_path)
            menu_photo_url = f"http://localhost:5000/uploads/{filename}"  # Correct URL

        # Insert the data into the database
        cursor = db.cursor()
        query = """
            INSERT INTO restaurants (title, restaurant_name, photo_url, rating, location, sub_location, special_item, description, recommendation, menu_photo_url, category)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (title, restaurant_name, photo_url, rating, location, sub_location, special_item, description, recommendation, menu_photo_url, category))
        db.commit()

        return jsonify({"message": "Restaurant added successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to get all restaurants
@app.route('/restaurants', methods=['GET'])
def get_all_restaurants():
    try:
        cursor = db.cursor(dictionary=True)  # Fetch results as dictionary
        cursor.execute("SELECT * FROM restaurants")
        restaurants = cursor.fetchall()  # Get all records

        # Return the restaurant data in JSON format
        return jsonify(restaurants), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to handle chat functionality (for example, use NLP for intent recognition)
# Load intents and model data
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json["message"]
    response = get_response(message)
    return jsonify({"response": response})

def get_response(msg):
    # Call the entity extraction function
    places, foods, adjectives = extract_entities(msg)

    # Print the extracted entities to the console
    print(f"\nEntities detected from message: '{msg}'")
    print("Detected places:", places)
    print("Detected foods:", foods)
    print("Detected adjectives:", adjectives)

    # Proceed with intent recognition
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."

if __name__ == "__main__":
    app.run(port=5000)
