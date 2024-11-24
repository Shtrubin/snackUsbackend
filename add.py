from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Folder to save uploaded images
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
    database="food_blog"
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
            INSERT INTO restaurants (title, restaurant_name, photo_url, rating, location, sub_location, special_item, description, recommendation, menu_photo_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (title, restaurant_name, photo_url, rating, location, sub_location, special_item, description, recommendation, menu_photo_url))
        db.commit()

        return jsonify({"message": "Restaurant added successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
