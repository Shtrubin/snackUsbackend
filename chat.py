import re
import torch
from nltk_utils import bag_of_words, tokenize, stem 
from entity_extraction import extract_entities
from model import NeuralNet
import json
import random
import mysql.connector
positive_adjectives = ['Best', 'Good', 'Excellent', 'Great', 'Amazing', 'Awesome', 'Fantastic']
negative_adjectives = ['Bad', 'Poor', 'Worst', 'Horrible', 'Terrible', 'Awful']
 
with open('intents.json', 'r',encoding='utf-8') as json_data:
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

def preprocess_message(msg):
    tokenized_msg = tokenize(msg)
    processed_msg = " ".join([stem(word) for word in tokenized_msg])
    return processed_msg

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
def extract_place_name(msg):
    match = re.search(r'\bin\s+([a-zA-Z0-9\s]+)', msg)
    if match:
        return match.group(1).strip() 
    return None

def get_response(msg, db):
    msg = preprocess_message(msg) 
    places, foods, adjectives = extract_entities(msg.title())

    print(f"\nEntities detected from message: '{msg}'")
    print("Detected places:", places)
    print("Detected foods:", foods)
    print("Detected adjectives:", adjectives)

    positive_detected = any(adj in positive_adjectives for adj in adjectives)
    negative_detected = any(adj in negative_adjectives for adj in adjectives)
    
    if positive_detected:
        return fetch_restaurants_by_rating(min_rating=8, db=db)
    elif negative_detected:
        return fetch_restaurants_by_rating(max_rating=3, db=db)
    
    if len(foods) > 0:
        return fetch_restaurants_by_food(foods[0], db)
    
    if len(places) == 0:
        location = search_location_in_database(msg, db)
        if location:
            return handle_location_response(location, db)
        else:
            return process_intent_with_model(msg, db)
    else:
        return process_location_based_query(places[0], db)

def search_location_in_database(msg, db=None):
    place_name = extract_place_name(msg)  
    if not place_name:
        return None  

    cursor = db.cursor()
    cursor.execute("SELECT DISTINCT location FROM restaurants WHERE location LIKE %s OR sub_location LIKE %s", 
                   ('%' + place_name + '%', '%' + place_name + '%'))
    result = cursor.fetchall()

    if result:
        return result[0][0] 
    return None 

def handle_location_response(location,db=None):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, rating, description FROM restaurants WHERE location = %s OR sub_location LIKE %s", (location, location))
    result = cursor.fetchall()

    if result:
        response = f"Here are some restaurants in {location}:\n"
        for row in result:
            response += f"{row[0]} - Rating: {row[1]}\nDescription: {row[2]}\n\n"
        return response
    return f"Sorry, I couldn't find any restaurants in {location}."

def fetch_restaurants_by_food(food, db=None):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, special_item, description FROM restaurants WHERE special_item LIKE %s OR recommendation LIKE %s", ('%' + food + '%', '%' + food + '%'))
    result = cursor.fetchall()

    if result:
        response = f"Here are some restaurants that serve {food}:\n"
        for row in result:
            response += f"{row[0]} - Special Item: {row[1]}\nDescription: {row[2]}\n\n"
        return response
    return f"Sorry, I couldn't find any restaurants serving {food}."

def fetch_restaurants_by_rating(min_rating=None, max_rating=None, db=None):
    cursor = db.cursor()

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

def process_intent_with_model(msg, db=None):
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
                        return fetch_restaurants_by_category(category, db)
                    else:
                        return "Could you please specify a category like 'local', 'mid-range', or 'high-end'?"

                elif tag == "find_by_special_item" and len(foods) > 0:
                    return fetch_restaurants_by_special_item(foods[0], db)

                elif tag == "find_by_rating":
                    rating = extract_rating(msg)
                    if rating:
                        return fetch_restaurants_rating(rating, db)
                    else:
                        return "Please specify a rating between 1 and 10."

                elif tag == "find_by_sub_location":
                    return fetch_restaurants_by_sub_location(msg,db)
                else:
                    return random.choice(intent['responses'])

        return "I do not understand..."

    return "I do not understand..."

def process_location_based_query(place, db=None):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, rating, description FROM restaurants WHERE location = %s OR sub_location LIKE %s", (place,place))
    result = cursor.fetchall()

    if result:
        response = f"Here are some great restaurants in {place}:\n"
        for row in result:
            response += f"{row[0]} - Rating: {row[1]}\nDescription: {row[2]}\n\n"
        return response
    else:
        return f"Sorry, I couldn't find any restaurants in {place}."

def fetch_restaurants_by_category(category, db=None):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, location, rating FROM restaurants WHERE category = %s", (category,))
    result = cursor.fetchall()

    if result:
        response = f"Here are some great restaurants in the {category} category:\n"
        for row in result:
            response += f"{row[0]} - {row[1]} - Rating: {row[2]}\n"
        return response
    return "Sorry, I couldn't find any restaurants in that category."

def fetch_restaurants_by_special_item(special_item, db=None):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, special_item, description FROM restaurants WHERE special_item LIKE %s", ('%' + special_item + '%',))
    result = cursor.fetchall()

    if result:
        response = f"Here are some restaurants that serve {special_item}:\n"
        for row in result:
            response += f"{row[0]} - Special Item: {row[1]}\nDescription: {row[2]}\n\n"
        return response
    return f"Sorry, I couldn't find any restaurants serving {special_item}."

def fetch_restaurants_rating(rating, db=None):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, rating FROM restaurants WHERE rating >= %s", (rating,))
    result = cursor.fetchall()

    if result:
        response = f"Here are some restaurants with a rating above {rating}:\n"
        for row in result:
            response += f"{row[0]} - Rating: {row[1]}\n"
        return response
    return f"Sorry, I couldn't find any restaurants with a rating above {rating}."

def fetch_restaurants_by_sub_location(msg, db=None):
    place_name = extract_place_name(msg)  
    if not place_name:
        return None  

    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, location, rating, description FROM restaurants WHERE sub_location LIKE %s", ('%' + place_name + '%',))
    result = cursor.fetchall()

    if result:
        response = f"Here are some restaurants in the {sub_location} sub-location:\n"
        for row in result:
            response += f"{row[0]} - Location: {row[1]} - Rating: {row[2]}\nDescription: {row[3]}\n\n"
        return response
    else:
        return f"Sorry, I couldn't find any restaurants in the {sub_location} sub-location."

