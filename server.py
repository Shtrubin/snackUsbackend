from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import random
import json
import mysql.connector
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from entity_extraction import extract_entities
from chat import get_response  

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

        return jsonify({
            "message": "Login successful!",
            "user_id": user['id'],
            "username": user['username']
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/restaurant/<int:id>', methods=['GET'])
def get_restaurant(id):
    try:
        cursor = db.cursor(dictionary=True)
        
        # Fetch restaurant details
        cursor.execute("""
            SELECT r.id, r.title, r.restaurant_name, r.photo_url, r.rating, r.location, r.special_item, r.description,
                   r.recommendation, r.menu_photo_url, r.category
            FROM restaurants r
            WHERE r.id = %s
        """, (id,))
        
        restaurant = cursor.fetchone()

        if not restaurant:
            return jsonify({"error": "Restaurant not found"}), 404

        # Fetch reviews for the restaurant
        cursor.execute("""
            SELECT rev.review_text, u.username
            FROM reviews rev
            LEFT JOIN users u ON rev.user_id = u.id
            WHERE rev.restaurant_id = %s
        """, (id,))
        
        reviews = cursor.fetchall()

        # If no reviews exist, set it as an empty list
        if not reviews:
            reviews = []

        # Add reviews to the restaurant data
        restaurant['reviews'] = reviews

        return jsonify(restaurant), 200

    except Exception as e:
        print(f"Error occurred: {e}")
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

@app.route('/submit_review', methods=['POST'])
def submit_review():
    try:
        data = request.json
        review_text = data['review_text']
        user_id = data['user_id']
        restaurant_id = data['restaurant_id']

        if not review_text or not user_id or not restaurant_id:
            return jsonify({"error": "Missing required fields"}), 400

        # Insert review into the database
        cursor = db.cursor()
        query = """
            INSERT INTO reviews (user_id, restaurant_id, review_text)
            VALUES (%s, %s, %s)
        """
        cursor.execute(query, (user_id, restaurant_id, review_text))
        db.commit()

        # Return a success message instead of username
        return jsonify({"message": "Review submitted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500






@app.route("/chat", methods=["POST"])
def chat():
    message = request.json["message"]
    response = get_response(message, db)  
    return jsonify({"response": response})  

if __name__ == "__main__":
    app.run(port=5000)
