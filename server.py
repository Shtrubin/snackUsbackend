from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import random
import json
import mysql.connector
import re
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from entity_extraction import extract_entities

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="snackus"
)

# Positive and Negative adjectives
positive_adjectives= ['Best', 'Good', 'Excellent', 'Great', 'Amazing', 'Awesome', 'Fantastic']
negative_adjectives= ['Bad', 'Poor', 'Worst', 'Horrible', 'Terrible', 'Awful']

@app.route('/register', methods=['POST'])
def register_user():
    try:
        data = request.json
        username = data['username']
        email = data['email']
        password = data['password']

        if not username or not email or not password:
            return jsonify({"error": "Please provide all required fields"}), 400

        # Check if the email is the admin email
        if email == "adminsnackus@gmail.com":
            return jsonify({"error": "This email is reserved for admin, registration not allowed."}), 400

        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()
        if existing_user:
            return jsonify({"error": "Email is already registered"}), 400

        hashed_password = generate_password_hash(password)

        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                       (username, email, hashed_password))
        db.commit()

        return jsonify({"message": "User registered successfully!"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login_user():
    try:
        data = request.json
        email = data['email']
        password = data['password']

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if not user or not check_password_hash(user['password'], password):
            return jsonify({"error": "Invalid email or password"}), 401

        return jsonify({"message": "Login successful!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/restaurant/<int:id>', methods=['GET'])
def get_restaurant(id):
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM restaurants WHERE id = %s", (id,))
        restaurant = cursor.fetchone()

        if not restaurant:
            return jsonify({"error": "Restaurant not found"}), 404

        return jsonify(restaurant), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/add_restaurant', methods=['POST'])
def add_restaurant():
    try:
        data = request.form
        title = data['title']
        restaurant_name = data['restaurant_name']
        rating = data['rating']
        location = data['location']
        sub_location = data['sub_location']
        special_item = data['special_item']
        description = data['description']
        recommendation = data['recommendation']
        category = data['category']

        if category not in ['local', 'mid-range', 'high-end']:
            return jsonify({"error": "Invalid category. Allowed values are 'local', 'mid-range', or 'high-end'."}), 400

        restaurant_photo = request.files['restaurant_photo']
        menu_photo = request.files['menu_photo']
        
        photo_url = None
        menu_photo_url = None

        if restaurant_photo and allowed_file(restaurant_photo.filename):
            filename = secure_filename(restaurant_photo.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            restaurant_photo.save(file_path)
            photo_url = f"http://localhost:5000/uploads/{filename}"

        if menu_photo and allowed_file(menu_photo.filename):
            filename = secure_filename(menu_photo.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            menu_photo.save(file_path)
            menu_photo_url = f"http://localhost:5000/uploads/{filename}"

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

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/restaurants', methods=['GET'])
def get_all_restaurants():
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM restaurants")
        restaurants = cursor.fetchall()

        return jsonify(restaurants), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

def extract_rating(msg):
    match = re.search(r"\b([1-9]|10)\b", msg)
    if match:
        return int(match.group(0))
    return None

def extract_category(msg):
    categories = ['local', 'mid-range', 'high-end']
    for category in categories:
        if category in msg.lower():
            return category
    return None

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json["message"]
    response = get_response(message)
    return jsonify({"response": response})

def get_response(msg):
    places, foods, adjectives = extract_entities(msg.title())

    print(f"\nEntities detected from message: '{msg}'")
    print("Detected places:", places)
    print("Detected foods:", foods)
    print("Detected adjectives:", adjectives)

    # Check for positive or negative adjectives in the message
    positive_detected = any(adj in positive_adjectives for adj in adjectives)
    negative_detected = any(adj in negative_adjectives for adj in adjectives)
    print("pos issss", positive_detected, negative_detected)

    # If positive adjective detected, fetch restaurants with high ratings (rating >= 8)
    if positive_detected:
        print("Positive adjective detected. Fetching restaurants with high ratings.")
        return fetch_restaurants_by_rating(min_rating=8)  # Fetch restaurants with rating 8 or above

    # If negative adjective detected, fetch restaurants with low ratings (rating <= 3)
    elif negative_detected:
        print("Negative adjective detected. Fetching restaurants with low ratings.")
        return fetch_restaurants_by_rating(max_rating=3)  # Fetch restaurants with rating 3 or below

    # If food items are mentioned, fetch restaurants by food
    if len(foods) > 0:
        return fetch_restaurants_by_food(foods[0])

    # If no food is mentioned, check for location-based queries
    if len(places) == 0:
        location = search_location_in_database(msg)
        if location:
            return handle_location_response(location)
        else:
            return process_intent_with_model(msg)
    else:
        return process_location_based_query(places[0])

def search_location_in_database(msg):
    cursor = db.cursor()
    cursor.execute("SELECT DISTINCT location FROM restaurants WHERE location LIKE %s", ('%' + msg + '%',))
    result = cursor.fetchall()

    if result:
        return result[0][0]
    return None

def handle_location_response(location):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, rating, description FROM restaurants WHERE location = %s", (location,))
    result = cursor.fetchall()

    if result:
        response = f"Here are some restaurants in {location}:\n"
        for row in result:
            response += f"{row[0]} - Rating: {row[1]}\nDescription: {row[2]}\n\n"
        return response
    return f"Sorry, I couldn't find any restaurants in {location}."

def fetch_restaurants_by_food(food):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, special_item, description FROM restaurants WHERE special_item LIKE %s OR recommendation LIKE %s", ('%' + food + '%', '%' + food + '%'))
    result = cursor.fetchall()

    if result:
        response = f"Here are some restaurants that serve {food}:\n"
        for row in result:
            response += f"{row[0]} - Special Item: {row[1]}\nDescription: {row[2]}\n\n"
        return response
    return f"Sorry, I couldn't find any restaurants serving {food}."

def fetch_restaurants_by_rating(min_rating=None, max_rating=None):
    cursor = db.cursor()

    # Adjust SQL query based on the provided rating thresholds
    if min_rating is not None and max_rating is None:
        cursor.execute("SELECT restaurant_name, rating FROM restaurants WHERE rating >= %s", (min_rating,))
    elif max_rating is not None and min_rating is None:
        cursor.execute("SELECT restaurant_name, rating FROM restaurants WHERE rating <= %s", (max_rating,))
    elif min_rating is not None and max_rating is not None:
        cursor.execute("SELECT restaurant_name, rating FROM restaurants WHERE rating BETWEEN %s AND %s", (min_rating, max_rating))
    else:
        return "Invalid rating parameters."

    result = cursor.fetchall()

    if result:
        response = f"Here are some restaurants with the requested rating:\n"
        for row in result:
            response += f"{row[0]} - Rating: {row[1]}\n"
        return response
    return "Sorry, I couldn't find any restaurants with the specified rating."

def process_intent_with_model(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.6:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "find_restaurant":
                    return "Sure! Where would you like to eat? Any preferences for location or category?"

                elif tag == "find_by_category":
                    category = extract_category(msg)
                    if category:
                        return fetch_restaurants_by_category(category)
                    else:
                        return "Could you please specify a category like 'local', 'mid-range', or 'high-end'?"

                elif tag == "find_by_special_item" and len(foods) > 0:
                    return fetch_restaurants_by_special_item(foods[0])

                elif tag == "find_by_rating":
                    rating = extract_rating(msg)
                    if rating:
                        return fetch_restaurants_rating(rating)
                    else:
                        return "Please specify a rating between 1 and 10."

                elif tag == "find_by_sub_location" and len(places) > 0:
                    return fetch_restaurants_by_sub_location(places[0])

        return "I do not understand..."

    return "I do not understand..."

def process_location_based_query(place):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, rating, description FROM restaurants WHERE location = %s", (place,))
    result = cursor.fetchall()

    if result:
        response = f"Here are some great restaurants in {place}:\n"
        for row in result:
            response += f"{row[0]} - Rating: {row[1]}\nDescription: {row[2]}\n\n"
        return response
    else:
        return f"Sorry, I couldn't find any restaurants in {place}."

def fetch_restaurants_by_category(category):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, location, rating FROM restaurants WHERE category = %s", (category,))
    result = cursor.fetchall()

    if result:
        response = f"Here are some great restaurants in the {category} category:\n"
        for row in result:
            response += f"{row[0]} - {row[1]} - Rating: {row[2]}\n"
        return response
    return "Sorry, I couldn't find any restaurants in that category."

def fetch_restaurants_by_special_item(special_item):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, special_item, description FROM restaurants WHERE special_item LIKE %s", ('%' + special_item + '%',))
    result = cursor.fetchall()

    if result:
        response = f"Here are some restaurants that serve {special_item}:\n"
        for row in result:
            response += f"{row[0]} - Special Item: {row[1]}\nDescription: {row[2]}\n\n"
        return response
    return f"Sorry, I couldn't find any restaurants serving {special_item}."

def fetch_restaurants_rating(rating):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, rating FROM restaurants WHERE rating >= %s", (rating,))
    result = cursor.fetchall()

    if result:
        response = f"Here are some restaurants with a rating above {rating}:\n"
        for row in result:
            response += f"{row[0]} - Rating: {row[1]}\n"
        return response
    return f"Sorry, I couldn't find any restaurants with a rating above {rating}."

if __name__ == "__main__":
    app.run(port=5000)
